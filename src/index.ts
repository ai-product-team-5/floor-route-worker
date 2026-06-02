import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { createClient } from '@libsql/client'
import { createHash, randomUUID } from 'node:crypto'
import { imageSize } from 'image-size'
import sharp from 'sharp'
import cv from '@techstark/opencv-js'
// imagePreprocess is no longer used — redraw and walls both use fixed 3:4 canvas

// --- Config from environment variables ---

const PORT = Number(process.env.PORT || 8787)
const DB_URL = process.env.DB_URL || 'file:./data/floor-route.db'
const VISION_MODEL_BASE_URL = process.env.VISION_MODEL_BASE_URL || ''
const VISION_MODEL_API_KEY = process.env.VISION_MODEL_API_KEY || ''
const VISION_MODEL_NAME = process.env.VISION_MODEL_NAME || ''
const IMAGE_MODEL_BASE_URL = process.env.IMAGE_MODEL_BASE_URL || ''
const IMAGE_MODEL_API_KEY = process.env.IMAGE_MODEL_API_KEY || ''
const IMAGE_MODEL_NAME = process.env.IMAGE_MODEL_NAME || ''

// --- Database setup ---

const db = createClient({ url: DB_URL })

async function initDb() {
  await db.executeMultiple(`
    CREATE TABLE IF NOT EXISTS api_keys (
      id TEXT PRIMARY KEY,
      key_hash TEXT UNIQUE NOT NULL,
      label TEXT,
      status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
      created_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS api_key_credit_accounts (
      api_key_id TEXT PRIMARY KEY,
      balance INTEGER NOT NULL DEFAULT 0,
      updated_at INTEGER NOT NULL,
      FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
    );
    CREATE TABLE IF NOT EXISTS api_key_credit_ledger (
      id TEXT PRIMARY KEY,
      api_key_id TEXT NOT NULL,
      type TEXT NOT NULL CHECK (type IN ('grant', 'consume', 'refund')),
      amount INTEGER NOT NULL,
      ref_id TEXT,
      created_at INTEGER NOT NULL,
      FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
    );
    CREATE TABLE IF NOT EXISTS generations (
      id TEXT PRIMARY KEY,
      api_key_id TEXT NOT NULL,
      endpoint TEXT NOT NULL,
      cost_credits INTEGER NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed', 'refunded')),
      error_message TEXT,
      created_at INTEGER NOT NULL,
      finished_at INTEGER,
      FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
    );
    CREATE TABLE IF NOT EXISTS tasks (
      id TEXT PRIMARY KEY,
      api_key_id TEXT NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
      result_image_url TEXT,
      error_message TEXT,
      created_at INTEGER NOT NULL,
      finished_at INTEGER,
      FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
    );
  `)
}

// --- Types ---

type ApiKeyContext = {
  id: string
  status: 'active' | 'disabled'
}

type DestinationCandidate = {
  id: string
  title: string
  subtitle?: string
  confidence: number
}

// --- Helpers ---

function nowUnix() {
  return Math.floor(Date.now() / 1000)
}

function newId(prefix: string) {
  return `${prefix}_${randomUUID()}`
}

function sha256Hex(input: string): string {
  return createHash('sha256').update(input).digest('hex')
}

function getBearerToken(header: string | undefined): string | null {
  if (!header) return null
  const parts = header.split(' ')
  if (parts.length !== 2 || parts[0] !== 'Bearer' || !parts[1]) return null
  return parts[1]
}

function isDataUrl(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^data:image\/(png|jpeg|jpg|webp);base64,/.test(value)
  )
}

function normalizeLimit(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 5
  return Math.min(10, Math.max(1, Math.floor(value)))
}

function validateCandidates(value: any, limit: number): DestinationCandidate[] {
  if (!Array.isArray(value)) return []
  return value
    .filter(
      (item) =>
        item &&
        typeof item.id === 'string' &&
        typeof item.title === 'string' &&
        typeof item.confidence === 'number' &&
        item.confidence >= 0 &&
        item.confidence <= 1
    )
    .slice(0, limit)
    .map((item) => ({
      id: item.id,
      title: item.title,
      subtitle: typeof item.subtitle === 'string' ? item.subtitle : undefined,
      confidence: item.confidence,
    }))
}



// --- Model calls ---

async function callVisionJson(prompt: string, imageDataUrl: string): Promise<any> {
  const response = await fetch(
    `${VISION_MODEL_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${VISION_MODEL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(120_000),
      body: JSON.stringify({
        model: VISION_MODEL_NAME,
        temperature: 0,
        max_tokens: 1000,
        response_format: { type: 'json_object' },
        messages: [
          {
            role: 'system',
            content: 'You are a strict JSON API. Only output valid JSON. No Markdown, no explanations.',
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageDataUrl } },
            ],
          },
        ],
      }),
    }
  )

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Vision model failed: ${response.status} ${text}`)
  }

  const data: any = await response.json()
  const content = data.choices?.[0]?.message?.content

  if (typeof content !== 'string') {
    throw new Error('Vision model returned empty content')
  }

  try {
    return JSON.parse(content)
  } catch {
    throw new Error(`Vision model returned invalid JSON: ${content}`)
  }
}



/**
 * 互相关平移对齐：把 mask 对齐到 input 的墙线。
 * 在低分辨率（workW px 宽）下暴力搜索最优 (dx, dy)，再按比例放大到 mask 分辨率。
 * 只做平移，不做缩放/旋转。
 */
async function alignMaskToInput(inputBuf: Buffer, maskBuf: Buffer, inputW: number, inputH: number): Promise<Buffer> {
  const maskMeta = await sharp(maskBuf).metadata()
  const maskW = maskMeta.width!
  const maskH = maskMeta.height!

  const workW = Math.min(400, inputW)
  const workH = Math.round(inputH * workW / inputW)

  const inputGray = await sharp(inputBuf).resize(workW, workH, { fit: 'fill' }).grayscale().raw().toBuffer()
  const maskGray  = await sharp(maskBuf) .resize(workW, workH, { fit: 'fill' }).grayscale().raw().toBuffer()

  const inputBin = new Uint8Array(workW * workH)
  const maskBin  = new Uint8Array(workW * workH)
  for (let i = 0; i < inputBin.length; i++) {
    inputBin[i] = inputGray[i] < 100 ? 1 : 0
  }
  for (let i = 0; i < maskBin.length; i++) {
    maskBin[i] = maskGray[i] < 100 ? 1 : 0
  }

  let inputDarkCount = 0
  let maskDarkCount  = 0
  for (let i = 0; i < inputBin.length; i++) {
    if (inputBin[i]) inputDarkCount++
  }
  for (let i = 0; i < maskBin.length; i++) {
    if (maskBin[i]) maskDarkCount++
  }

  if (inputDarkCount < 100 || maskDarkCount < 100) {
    console.warn('Alignment skipped: too few dark pixels')
    return maskBuf
  }

  let bestScore = -1
  let bestDx = 0
  let bestDy = 0

  for (let dy = -15; dy <= 15; dy++) {
    for (let dx = -15; dx <= 15; dx++) {
      let score = 0
      for (let y = 0; y < workH; y += 2) {
        for (let x = 0; x < workW; x += 2) {
          const x2 = x - dx, y2 = y - dy
          if (x2 < 0 || x2 >= workW || y2 < 0 || y2 >= workH) continue
          if (inputBin[y * workW + x] && maskBin[y2 * workW + x2]) score++
        }
      }
      if (score > bestScore) { bestScore = score; bestDx = dx; bestDy = dy }
    }
  }

  const confidence = bestScore / Math.max(1, inputDarkCount)
  if (confidence < 0.05) {
    console.warn(`Alignment skipped: low confidence ${confidence.toFixed(3)}`)
    return maskBuf
  }

  const shiftX = Math.round(bestDx * maskW / workW)
  const shiftY = Math.round(bestDy * maskH / workH)

  if (Math.abs(bestDx) === 15 || Math.abs(bestDy) === 15) {
    console.warn(`Alignment hit search boundary: dx=${bestDx} dy=${bestDy}`)
  }

  console.log(`Alignment: dx=${bestDx} dy=${bestDy} score=${bestScore} confidence=${confidence.toFixed(3)} shiftX=${shiftX} shiftY=${shiftY} applied=true`)

  if (shiftX === 0 && shiftY === 0) return maskBuf

  try {
    // Sharp applies extract BEFORE extend in its pipeline regardless of call order,
    // so we MUST split into two separate sharp() invocations to force the correct order.
    const extendedBuf = await sharp(maskBuf)
      .extend({
        left:   Math.max(0,  shiftX),
        right:  Math.max(0, -shiftX),
        top:    Math.max(0,  shiftY),
        bottom: Math.max(0, -shiftY),
        background: { r: 255, g: 255, b: 255, alpha: 1 },
      })
      .png()
      .toBuffer()

    const alignedBuf = await sharp(extendedBuf)
      .extract({
        left:   Math.max(0, -shiftX),
        top:    Math.max(0, -shiftY),
        width:  maskW,
        height: maskH,
      })
      .png()
      .toBuffer()

    const outMeta = await sharp(alignedBuf).metadata()
    if (outMeta.width !== maskW || outMeta.height !== maskH) {
      console.warn(`Alignment output size mismatch (got ${outMeta.width}x${outMeta.height}, expected ${maskW}x${maskH}), returning unaligned mask`)
      return maskBuf
    }

    return alignedBuf
  } catch (err) {
    console.warn(`Alignment shift failed (${err instanceof Error ? err.message : String(err)}), returning unaligned mask`)
    return maskBuf
  }
}

/**
 * Score alignment quality by counting overlapping dark pixels at work resolution.
 * Both buffers resized to (workW, workH), grayscale, binarized at threshold 100.
 * Higher score = better alignment. Used by dispatchAlignment to pick the best candidate.
 */
async function darkPixelOverlapScore(
  inputBuf: Buffer,
  candidateBuf: Buffer,
  workW: number,
  workH: number,
): Promise<number> {
  const [inputRaw, candidateRaw] = await Promise.all([
    sharp(inputBuf).resize(workW, workH, { fit: 'fill' }).grayscale().raw().toBuffer(),
    sharp(candidateBuf).resize(workW, workH, { fit: 'fill' }).grayscale().raw().toBuffer(),
  ])
  let score = 0
  for (let i = 0; i < inputRaw.length; i++) {
    if (inputRaw[i] < 100 && candidateRaw[i] < 100) score++
  }
  return score
}



/**
 * Align mask to photo via ECC (Enhanced Correlation Coefficient) on
 * distance-transformed edge maps. Tries HOMOGRAPHY first, then EUCLIDEAN
 * as fallback. Returns the warped mask at photo native resolution.
 *
 * Wave 1 probe verified convergence on the canonical pair: HOMOGRAPHY ecc≈0.84
 * @ 800×559 work resolution (~2.4s), EUCLIDEAN ecc≈0.82 (~0.2s). See
 * `.omo/notepads/ecc-alignment/learnings.md`.
 */
async function alignMaskToInputECC(
  inputBuf: Buffer,
  maskBuf: Buffer,
  inputW: number,
  inputH: number,
): Promise<{
  aligned: Buffer
  method: 'ecc-homography' | 'ecc-euclidean' | 'none'
  reason?: string
  warpMatrix?: number[][]
  ecc?: number
}> {
  // Mask native dims (do NOT assume mask is at photo dims — caller may pass a
  // mask of arbitrary size, e.g. 2382×1664 from a 1400×978 photo). Read for
  // logging; sharp().resize handles any size mismatch implicitly below.
  const maskMeta = await sharp(maskBuf).metadata()
  const maskW = maskMeta.width!
  const maskH = maskMeta.height!

  // Work resolution (cap at 800px wide, proportional height)
  const workW = Math.min(800, inputW)
  const workH = Math.round(inputH * workW / inputW)

  // Declare every cv.Mat upfront so finally cleanup can always reference them.
  let photoRgba: cv.Mat | null = null
  let photoGray: cv.Mat | null = null
  let photoCanny: cv.Mat | null = null
  let photoCannyInv: cv.Mat | null = null
  let photoDt: cv.Mat | null = null
  let photoDistN: cv.Mat | null = null
  let maskRgba: cv.Mat | null = null
  let maskGray: cv.Mat | null = null
  let maskBin: cv.Mat | null = null
  let kernel: cv.Mat | null = null
  let maskDilated: cv.Mat | null = null
  let maskCanny: cv.Mat | null = null
  let maskCannyInv: cv.Mat | null = null
  let maskDt: cv.Mat | null = null
  let maskDistN: cv.Mat | null = null
  let warpH: cv.Mat | null = null
  let warpE: cv.Mat | null = null
  let warpFull: cv.Mat | null = null
  let maskMatFull: cv.Mat | null = null
  let warpedMat: cv.Mat | null = null

  try {
    // Resize photo and mask to work resolution as RGBA for cv ingestion.
    const [photoRaw, maskRaw] = await Promise.all([
      sharp(inputBuf).resize(workW, workH, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true }),
      sharp(maskBuf).resize(workW, workH, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true }),
    ])
    photoRgba = cv.matFromArray(workH, workW, cv.CV_8UC4, photoRaw.data)
    maskRgba = cv.matFromArray(workH, workW, cv.CV_8UC4, maskRaw.data)

    // Photo: gray → Canny(50,150) → invert → distanceTransform → normalize
    photoGray = new cv.Mat()
    cv.cvtColor(photoRgba, photoGray, cv.COLOR_RGBA2GRAY)
    photoCanny = new cv.Mat()
    cv.Canny(photoGray, photoCanny, 50, 150)
    photoCannyInv = new cv.Mat()
    cv.bitwise_not(photoCanny, photoCannyInv)
    photoDt = new cv.Mat()
    cv.distanceTransform(photoCannyInv, photoDt, cv.DIST_L2, 5)
    photoDistN = new cv.Mat()
    cv.normalize(photoDt, photoDistN, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_32F)

    // Mask: gray → threshold(127) inverse → dilate(3×3, 1 iter) → Canny → invert
    //       → distanceTransform → normalize. Walls in the mask are dark (low
    //       grayscale) so THRESH_BINARY_INV makes wall pixels = 255 before edge
    //       detection.
    maskGray = new cv.Mat()
    cv.cvtColor(maskRgba, maskGray, cv.COLOR_RGBA2GRAY)
    maskBin = new cv.Mat()
    cv.threshold(maskGray, maskBin, 127, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))
    maskDilated = new cv.Mat()
    cv.dilate(maskBin, maskDilated, kernel, new cv.Point(-1, -1), 1)
    maskCanny = new cv.Mat()
    cv.Canny(maskDilated, maskCanny, 50, 150)
    maskCannyInv = new cv.Mat()
    cv.bitwise_not(maskCanny, maskCannyInv)
    maskDt = new cv.Mat()
    cv.distanceTransform(maskCannyInv, maskDt, cv.DIST_L2, 5)
    maskDistN = new cv.Mat()
    cv.normalize(maskDt, maskDistN, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_32F)

    // --- Try HOMOGRAPHY first (criteria 100, 1e-4, gaussFiltSize=5 default) ---
    warpH = cv.Mat.eye(3, 3, cv.CV_32F)
    const critH = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 100, 1e-4)
    let homoEcc = -1
    let homoOk = false
    let homoReason = ''
    const tHomo = Date.now()
    try {
      // 6-arg overload: omit inputMask + gaussFiltSize. gaussFiltSize defaults
      // to 5 on the OpenCV side, matching the probe.
      homoEcc = cv.findTransformECC(photoDistN, maskDistN, warpH, cv.MOTION_HOMOGRAPHY, critH, new cv.Mat(), 5)
      homoOk = true
    } catch (err) {
      homoReason = classifyEccError(err)
    }
    const homoMs = Date.now() - tHomo

    let methodChosen: 'ecc-homography' | 'ecc-euclidean' | 'none' = 'none'
    let chosenEcc = -1
    let chosenDet = 0
    let chosenRotDeg = 0
    let chosenTx = 0
    let chosenTy = 0
    let chosenMs = 0
    let chosenWarp2D: number[][] = []
    let chosenIs3x3 = true

    if (homoOk) {
      // Validate 3×3 warp: extract det, rotDeg, tx, ty from CV_32F row-major.
      const wd = warpH.data32F
      const h00 = wd[0], h01 = wd[1], h02 = wd[2]
      const h10 = wd[3], h11 = wd[4], h12 = wd[5]
      const h20 = wd[6], h21 = wd[7], h22 = wd[8]
      const det = h00 * h11 - h01 * h10
      const rotDeg = Math.atan2(h10, h00) * 180 / Math.PI
      const tx = h02
      const ty = h12

      if (det < 0.7 || det > 1.3) {
        homoOk = false
        homoReason = `degenerate det=${det.toFixed(2)}`
      } else if (Math.abs(rotDeg) >= 5) {
        homoOk = false
        homoReason = `degenerate rot=${rotDeg.toFixed(2)}`
      } else if (Math.abs(tx) + Math.abs(ty) >= 100) {
        homoOk = false
        homoReason = `degenerate trans=${(Math.abs(tx) + Math.abs(ty)).toFixed(1)}`
      } else {
        methodChosen = 'ecc-homography'
        chosenEcc = homoEcc
        chosenDet = det
        chosenRotDeg = rotDeg
        chosenTx = tx
        chosenTy = ty
        chosenMs = homoMs
        chosenIs3x3 = true
        chosenWarp2D = [
          [h00, h01, h02],
          [h10, h11, h12],
          [h20, h21, h22],
        ]
      }
    }

    // --- Fallback: EUCLIDEAN (criteria 50, 1e-3) ---
    if (methodChosen === 'none') {
      warpE = cv.Mat.eye(2, 3, cv.CV_32F)
      const critE = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 50, 1e-3)
      let euEcc = -1
      let euOk = false
      let euReason = homoReason
      const tEu = Date.now()
      try {
        euEcc = cv.findTransformECC(photoDistN, maskDistN, warpE, cv.MOTION_EUCLIDEAN, critE, new cv.Mat(), 5)
        euOk = true
      } catch (err) {
        euReason = classifyEccError(err)
      }
      const euMs = Date.now() - tEu

      if (euOk) {
        // Validate 2×3 affine — same det/rot/translation gates as 3×3.
        // Conceptually equivalent to appending [0,0,1] row before checking.
        const ed = warpE.data32F
        const a00 = ed[0], a01 = ed[1], a02 = ed[2]
        const a10 = ed[3], a11 = ed[4], a12 = ed[5]
        const det = a00 * a11 - a01 * a10
        const rotDeg = Math.atan2(a10, a00) * 180 / Math.PI
        const tx = a02
        const ty = a12

        if (det < 0.7 || det > 1.3) {
          euReason = `degenerate det=${det.toFixed(2)}`
        } else if (Math.abs(rotDeg) >= 5) {
          euReason = `degenerate rot=${rotDeg.toFixed(2)}`
        } else if (Math.abs(tx) + Math.abs(ty) >= 100) {
          euReason = `degenerate trans=${(Math.abs(tx) + Math.abs(ty)).toFixed(1)}`
        } else {
          methodChosen = 'ecc-euclidean'
          chosenEcc = euEcc
          chosenDet = det
          chosenRotDeg = rotDeg
          chosenTx = tx
          chosenTy = ty
          chosenMs = euMs
          chosenIs3x3 = false
          chosenWarp2D = [
            [a00, a01, a02],
            [a10, a11, a12],
          ]
        }
      }

      if (methodChosen === 'none') {
        const reason = euReason || homoReason || 'noConv'
        console.log(`ECC: failed (reason: ${reason})`)
        return { aligned: maskBuf, method: 'none', reason }
      }
    }

    // --- Apply warp at full input resolution ---
    // Resize mask to photo dims first so the output matches photo dims (caller
    // contract is identical to alignMaskToInput).
    const { data: maskFullData, info: maskFullInfo } = await sharp(maskBuf)
      .resize(inputW, inputH, { fit: 'fill' })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true })
    maskMatFull = cv.matFromArray(maskFullInfo.height, maskFullInfo.width, cv.CV_8UC4, maskFullData)

    // Scale work-res warp to full-res. For nearly-identity matrices (det≈1,
    // rot<2°) the cross-scale factors on h01/h10 are negligible, so we only
    // scale the translation column and the perspective row per spec.
    const sx = inputW / workW
    const sy = inputH / workH

    warpedMat = new cv.Mat()
    if (chosenIs3x3) {
      const r0 = chosenWarp2D[0]
      const r1 = chosenWarp2D[1]
      const r2 = chosenWarp2D[2]
      warpFull = cv.matFromArray(3, 3, cv.CV_32F, [
        r0[0],      r0[1],      r0[2] * sx,
        r1[0],      r1[1],      r1[2] * sy,
        r2[0] / sx, r2[1] / sy, r2[2],
      ])
      cv.warpPerspective(
        maskMatFull,
        warpedMat,
        warpFull,
        new cv.Size(inputW, inputH),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Scalar(255, 255, 255, 255),
      )
    } else {
      const r0 = chosenWarp2D[0]
      const r1 = chosenWarp2D[1]
      warpFull = cv.matFromArray(2, 3, cv.CV_32F, [
        r0[0], r0[1], r0[2] * sx,
        r1[0], r1[1], r1[2] * sy,
      ])
      cv.warpAffine(
        maskMatFull,
        warpedMat,
        warpFull,
        new cv.Size(inputW, inputH),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Scalar(255, 255, 255, 255),
      )
    }

    const warpedBuf = await sharp(Buffer.from(warpedMat.data), {
      raw: { width: inputW, height: inputH, channels: 4 },
    })
      .png()
      .toBuffer()

    console.log(
      `ECC: method=${methodChosen} ecc=${chosenEcc.toFixed(3)} det=${chosenDet.toFixed(3)} rotDeg=${chosenRotDeg.toFixed(2)} tx=${chosenTx.toFixed(1)} ty=${chosenTy.toFixed(1)} ms=${chosenMs} maskNative=${maskW}x${maskH}`,
    )

    return {
      aligned: warpedBuf,
      method: methodChosen,
      ecc: chosenEcc,
      warpMatrix: chosenWarp2D,
    }
  } finally {
    const toDelete: Array<{ delete: () => void } | null | undefined> = [
      photoRgba, photoGray, photoCanny, photoCannyInv, photoDt, photoDistN,
      maskRgba, maskGray, maskBin, kernel, maskDilated,
      maskCanny, maskCannyInv, maskDt, maskDistN,
      warpH, warpE, warpFull,
      maskMatFull, warpedMat,
    ]
    for (const obj of toDelete) {
      try { obj?.delete() } catch {}
    }
  }
}

/**
 * Map an opencv-js exception to a stable reason tag. Emscripten may throw
 * either an `Error`, a numeric pointer, or an object with `.msg`. We probe
 * the message for the canonical "did not converge" / StsNoConv strings and
 * return 'noConv' if matched, otherwise 'error'.
 */
function classifyEccError(err: unknown): string {
  let msg = ''
  if (err instanceof Error) {
    msg = err.message
  } else if (typeof err === 'object' && err !== null) {
    const cand = err as { msg?: unknown; message?: unknown }
    if (typeof cand.msg === 'string') msg = cand.msg
    else if (typeof cand.message === 'string') msg = cand.message
    else msg = String(err)
  } else {
    msg = String(err)
  }
  return /StsNoConv|did not converge|not converged|noConv/i.test(msg) ? 'noConv' : 'error'
}

/**
 * Run all three alignment candidates, score each, return the best.
 * Scores via darkPixelOverlapScore at work resolution (400px wide).
 * Tie-break: translation > ecc > raw (most conservative wins).
 */
async function dispatchAlignment(
  inputBuf: Buffer,
  maskBuf: Buffer,
  inputW: number,
  inputH: number,
): Promise<Buffer> {
  try {
    const workW = Math.min(400, inputW)
    const workH = Math.round(inputH * workW / inputW)

    // Score raw mask as baseline
    const scoreRaw = await darkPixelOverlapScore(inputBuf, maskBuf, workW, workH)

    // Try ECC alignment
    let scoreEcc = -1
    let eccBuf: Buffer = maskBuf
    const eccResult = await alignMaskToInputECC(inputBuf, maskBuf, inputW, inputH)
    if (eccResult.method === 'ecc-homography' || eccResult.method === 'ecc-euclidean') {
      scoreEcc = await darkPixelOverlapScore(inputBuf, eccResult.aligned, workW, workH)
      eccBuf = eccResult.aligned
    }

    // Try translation
    const transBuf = await alignMaskToInput(inputBuf, maskBuf, inputW, inputH)
    const scoreTrans = await darkPixelOverlapScore(inputBuf, transBuf, workW, workH)

    // Pick winner (tie-break: trans > ecc > raw)
    let winner: 'raw' | 'trans' | 'ecc'
    let resultBuf: Buffer
    if (scoreTrans >= scoreEcc && scoreTrans >= scoreRaw) {
      winner = 'trans'; resultBuf = transBuf
    } else if (scoreEcc >= scoreRaw) {
      winner = 'ecc'; resultBuf = eccBuf
    } else {
      winner = 'raw'; resultBuf = maskBuf
    }

    console.log(`Dispatch: scoreRaw=${scoreRaw} scoreTrans=${scoreTrans} scoreEcc=${scoreEcc} winner=${winner}`)
    return resultBuf
  } catch (err) {
    console.warn(`Dispatch failed: ${err instanceof Error ? err.message : String(err)}; returning raw mask`)
    return maskBuf
  }
}

/**
 * 从 data URL 解析图片宽高，使用 image-size 库正确处理各种 JPEG/PNG 格式。
 */
function parseImageDimensions(dataUrl: string): { width: number; height: number } | null {
  try {
    const base64Match = dataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
    if (!base64Match) return null
    const buf = Buffer.from(base64Match[1], 'base64')
    const result = imageSize(buf)
    if (result.width && result.height) {
      return { width: result.width, height: result.height }
    }
    return null
  } catch {
    return null
  }
}

async function callImageModel(prompt: string, imageDataUrl: string, aspectRatio: string): Promise<string> {
  const maxRetries = 2
  let lastError: Error | null = null

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await callImageModelOnce(prompt, imageDataUrl, aspectRatio)
    } catch (error: any) {
      lastError = error
      const msg = error?.message || ''
      // Only retry on network-level failures, not on 4xx API errors
      const isNetworkError = msg === 'fetch failed' || msg.includes('ECONNRESET') || msg.includes('ETIMEDOUT') || msg.includes('UND_ERR')
      if (!isNetworkError || attempt === maxRetries - 1) throw error
      console.error(`Model request failed (attempt ${attempt + 1}/${maxRetries}): ${msg}, retrying...`)
      await new Promise(r => setTimeout(r, 3000))
    }
  }
  throw lastError!
}

async function callImageModelOnce(prompt: string, imageDataUrl: string, aspectRatio: string): Promise<string> {
  const response = await fetch(
    `${IMAGE_MODEL_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${IMAGE_MODEL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(480_000),
      body: JSON.stringify({
        model: IMAGE_MODEL_NAME,
        modalities: ['image', 'text'],
        image_config: {
          aspect_ratio: aspectRatio,
          image_size: '2K',
        },
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageDataUrl } },
            ],
          },
        ],
      }),
    }
  )

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Image model failed: ${response.status} ${text}`)
  }

  const data: any = await response.json()
  const message = data.choices?.[0]?.message

  let outputDataUrl: string | null = null
  const imageUrl = message?.images?.[0]?.image_url?.url
  if (typeof imageUrl === 'string' && imageUrl.startsWith('data:')) {
    outputDataUrl = imageUrl
  }
  if (!outputDataUrl && Array.isArray(message?.content)) {
    for (const part of message.content) {
      if (part.type === 'image_url' && part.image_url?.url?.startsWith('data:')) {
        outputDataUrl = part.image_url.url
        break
      }
    }
  }
  // Fallback: extract HTTP image URL from markdown or plain text content
  if (!outputDataUrl) {
    const contentStr = typeof message?.content === 'string' ? message.content : ''
    const urlMatch = contentStr.match(/https?:\/\/[^\s)]+\.(?:png|jpg|jpeg|webp)/i)
    if (urlMatch) {
      const imgResponse = await fetch(urlMatch[0], { signal: AbortSignal.timeout(60_000) })
      if (imgResponse.ok) {
        const imgBuf = Buffer.from(await imgResponse.arrayBuffer())
        const contentType = imgResponse.headers.get('content-type') || 'image/png'
        outputDataUrl = `data:${contentType};base64,${imgBuf.toString('base64')}`
      }
    }
  }
  if (!outputDataUrl) {
    throw new Error('Image model did not return a generated image')
  }
  return outputDataUrl
}

// --- Database operations ---

async function getApiKey(keyHash: string): Promise<ApiKeyContext | null> {
  const result = await db.execute({ sql: 'SELECT id, status FROM api_keys WHERE key_hash = ? LIMIT 1', args: [keyHash] })
  const row = result.rows[0]
  if (!row) return null
  return { id: row.id as string, status: row.status as 'active' | 'disabled' }
}

async function getBalance(apiKeyId: string): Promise<number> {
  const result = await db.execute({ sql: 'SELECT balance FROM api_key_credit_accounts WHERE api_key_id = ? LIMIT 1', args: [apiKeyId] })
  const row = result.rows[0]
  return (row?.balance as number) ?? 0
}

async function createGeneration(apiKeyId: string, endpoint: string): Promise<string> {
  const id = newId('gen')
  const now = nowUnix()
  await db.execute({ sql: 'INSERT INTO generations (id, api_key_id, endpoint, cost_credits, status, created_at) VALUES (?, ?, ?, 1, \'pending\', ?)', args: [id, apiKeyId, endpoint, now] })
  return id
}

async function deductOneCredit(apiKeyId: string, generationId: string): Promise<boolean> {
  const now = nowUnix()
  const result = await db.execute({ sql: 'UPDATE api_key_credit_accounts SET balance = balance - 1, updated_at = ? WHERE api_key_id = ? AND balance >= 1', args: [now, apiKeyId] })
  if (result.rowsAffected !== 1) return false
  await db.execute({ sql: 'INSERT INTO api_key_credit_ledger (id, api_key_id, type, amount, ref_id, created_at) VALUES (?, ?, \'consume\', -1, ?, ?)', args: [newId('ledger'), apiKeyId, generationId, now] })
  return true
}

async function markSucceeded(generationId: string) {
  await db.execute({ sql: 'UPDATE generations SET status = \'succeeded\', finished_at = ? WHERE id = ?', args: [nowUnix(), generationId] })
}

async function markFailedAndRefund(apiKeyId: string, generationId: string, errorMsg: string) {
  const now = nowUnix()
  await db.batch([
    { sql: 'UPDATE generations SET status = \'failed\', error_message = ?, finished_at = ? WHERE id = ?', args: [errorMsg.slice(0, 1000), now, generationId] },
    { sql: 'UPDATE api_key_credit_accounts SET balance = balance + 1, updated_at = ? WHERE api_key_id = ?', args: [now, apiKeyId] },
    { sql: 'INSERT INTO api_key_credit_ledger (id, api_key_id, type, amount, ref_id, created_at) VALUES (?, ?, \'refund\', 1, ?, ?)', args: [newId('ledger'), apiKeyId, generationId, now] },
  ])
}

async function markFailedNoRefund(generationId: string, errorMsg: string) {
  await db.execute({ sql: 'UPDATE generations SET status = \'failed\', error_message = ?, finished_at = ? WHERE id = ?', args: [errorMsg.slice(0, 1000), nowUnix(), generationId] })
}

// --- Hono app ---

const app = new Hono<{ Variables: { apiKey: ApiKeyContext } }>()

app.use('/api/*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}))

function jsonError(c: any, status: number, error: string, message: string) {
  return c.json({ error, message }, status)
}

app.use('/api/*', async (c, next) => {
  const token = getBearerToken(c.req.header('Authorization'))
  if (!token) return jsonError(c, 401, 'missing_api_key', 'API Key is not set.')

  const keyHash = sha256Hex(token)
  const row = await getApiKey(keyHash)
  if (!row) return jsonError(c, 401, 'invalid_api_key', 'Invalid API Key.')
  if (row.status !== 'active') return jsonError(c, 403, 'disabled_api_key', 'API Key is disabled.')

  c.set('apiKey', row)
  await next()
})

async function chargeAndRun(c: any, endpoint: string, fn: () => Promise<any>) {
  const apiKey = c.get('apiKey') as ApiKeyContext
  const generationId = await createGeneration(apiKey.id, endpoint)
  const deducted = await deductOneCredit(apiKey.id, generationId)

  if (!deducted) {
    await markFailedNoRefund(generationId, 'insufficient_credits')
    return jsonError(c, 402, 'insufficient_credits', 'Insufficient credits.')
  }

  try {
    const result = await fn()
    await markSucceeded(generationId)
    return c.json(result)
  } catch (error: any) {
    console.error('Model request failed:', error?.message || error)
    await markFailedAndRefund(apiKey.id, generationId, error?.message || 'unknown')
    return jsonError(c, 500, 'model_request_failed', `Model request failed: ${error?.message || 'unknown'}`)
  }
}

app.get('/', (c) => {
  return c.json({
    message: 'FloorRoute API is running.',
    routes: [
      'GET /api/credits',
      'POST /api/search',
      'POST /api/walls',
      'POST /api/endpoints',
      'GET /api/task/:id',
    ],
  })
})

app.get('/api/credits', async (c) => {
  const apiKey = c.get('apiKey') as ApiKeyContext
  return c.json({ balance: await getBalance(apiKey.id) })
})

app.post('/api/search', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }
  if (typeof body.query !== 'string' || body.query.trim() === '') {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing query.')
  }

  const query = body.query.trim()
  const limit = normalizeLimit(body.limit)

  return chargeAndRun(c, '/api/search', async () => {
    const result = await callVisionJson(
      `你是一个室内平面图目的地搜索引擎。

用户提供一张室内平面图和搜索词："${query}"。

搜索规则：
1. 仅根据图中可见的文字、房间名称、设施标注进行匹配。
2. 支持同义词和模糊匹配（如"厕所"匹配"卫生间"/"洗手间"/"WC"，"电梯"匹配"升降机"/"Elevator"）。
3. 绝不编造图中不存在的地点。
4. 最多返回 ${limit} 个候选结果，按匹配置信度从高到低排序。
5. id 使用英文小写加连字符，基于位置特征生成（如 "restroom-east"、"elevator-b1"）。
6. subtitle 填写该地点在图中的相对位置描述（如"东侧电梯旁"、"A区走廊尽头"）。

输出格式（仅输出 JSON）：
{"message":"找到匹配的目的地。","candidates":[{"id":"restroom-east","title":"卫生间","subtitle":"东侧电梯旁","confidence":0.92}]}
无匹配时返回：{"message":"未找到匹配的目的地。","candidates":[]}`,
      body.imageDataUrl
    )

    const candidates = validateCandidates(result.candidates, limit)
    return {
      candidates,
      message: typeof result.message === 'string'
        ? result.message
        : candidates.length > 0 ? 'Found matching destinations.' : 'No matching destination found.',
    }
  })
})

// --- Async floor plan redraw ---

app.post('/api/redraw', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }
  const apiKey = c.get('apiKey') as ApiKeyContext
  const generationId = await createGeneration(apiKey.id, '/api/redraw')
  const deducted = await deductOneCredit(apiKey.id, generationId)
  if (!deducted) {
    await markFailedNoRefund(generationId, 'insufficient_credits')
    return jsonError(c, 402, 'insufficient_credits', 'Insufficient credits.')
  }
  const taskId = newId('task')
  await db.execute({
    sql: 'INSERT INTO tasks (id, api_key_id, status, created_at) VALUES (?, ?, \'processing\', ?)',
    args: [taskId, apiKey.id, nowUnix()],
  })
  void processRedrawTask(taskId, apiKey.id, generationId, body.imageDataUrl)
  return c.json({ taskId, message: 'Floor plan redraw started.' })
})

// --- Async wall mask generation ---

app.post('/api/walls', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }

  const apiKey = c.get('apiKey') as ApiKeyContext

  // Charge credit upfront
  const generationId = await createGeneration(apiKey.id, '/api/walls')
  const deducted = await deductOneCredit(apiKey.id, generationId)
  if (!deducted) {
    await markFailedNoRefund(generationId, 'insufficient_credits')
    return jsonError(c, 402, 'insufficient_credits', 'Insufficient credits.')
  }

  // Create task
  const taskId = newId('task')
  await db.execute({
    sql: 'INSERT INTO tasks (id, api_key_id, status, created_at) VALUES (?, ?, \'processing\', ?)',
    args: [taskId, apiKey.id, nowUnix()],
  })

  // Start background processing (fire and forget)
  void processWallsTask(taskId, apiKey.id, generationId, body.imageDataUrl)

  // Return immediately
  return c.json({ taskId, message: 'Wall mask generation started.' })
})

async function processWallsTask(
  taskId: string,
  apiKeyId: string,
  generationId: string,
  imageDataUrl: string,
) {
  try {
    const wallsPrompt = `将这张建筑平面图转换为纯二值墙体掩码图。图中墙体已经是完整的黑色线条，你只需要删除非墙体元素。

严格输出规则：
- 仅使用两种颜色：纯黑 (#000000) 和纯白 (#FFFFFF)
- 零容忍灰色像素、抗锯齿、渐变或任何中间值
- 黑色 = 墙体线条。保留输入图中已有的墙体线条，位置和粗细不变
- 白色 = 所有可通行空间（走廊、房间内部、室外区域、门洞）
- 门的开口（缺口、弧线、开向标识）必须渲染为白色，以保持房间与走廊的连通性
- 门的弧线不要绘制，涂为白色
- 将所有非墙体元素涂为白色：文字、标注、数字、门的弧线、窗户标记、家具图标
- 输出图像必须与输入图像尺寸完全一致
- 不得移动、添加或修改任何墙体线条的位置

不要输出任何文字，只输出掩码图像。`

    const rawOutput = await callImageModel(wallsPrompt, imageDataUrl, '3:4')
    let wallMaskDataUrl = rawOutput

    // Resize mask to match input dimensions if needed, then align via cross-correlation
    const inputBase64Match = imageDataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
    if (inputBase64Match) {
      const inputBuf = Buffer.from(inputBase64Match[1], 'base64')
      const maskBase64Match = wallMaskDataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
      if (maskBase64Match) {
        const maskBuf = Buffer.from(maskBase64Match[1], 'base64')
        const inputDims = parseImageDimensions(imageDataUrl)
        if (inputDims) {
          const alignedBuf = await dispatchAlignment(inputBuf, maskBuf, inputDims.width, inputDims.height)
          wallMaskDataUrl = `data:image/png;base64,${alignedBuf.toString('base64')}`
        }
      }
    }

    await markSucceeded(generationId)
    await db.execute({
      sql: 'UPDATE tasks SET status = \'completed\', result_image_url = ?, finished_at = ? WHERE id = ?',
      args: [wallMaskDataUrl, nowUnix(), taskId],
    })
  } catch (error: any) {
    const errorMsg = error?.message || 'unknown'
    console.error('Wall mask generation failed:', errorMsg)
    await markFailedAndRefund(apiKeyId, generationId, errorMsg)
    await db.execute({
      sql: 'UPDATE tasks SET status = \'failed\', error_message = ?, finished_at = ? WHERE id = ?',
      args: [errorMsg.slice(0, 1000), nowUnix(), taskId],
    })
  }
}

async function processRedrawTask(taskId: string, apiKeyId: string, generationId: string, imageDataUrl: string) {
  try {
    const redrawPrompt = `将输入的平面图照片重绘为干净的标准建筑平面图。

输出画布为 3:4 竖版比例。保持原图中各结构的比例关系不变，将平面图完整绘制在画布中央，不得截断任何内容。

要求：
- 纯白背景、黑色线条，墙体用较粗实线，其他元素用细线
- 修正透视畸变，确保墙线横平竖直
- 去除拍摄痕迹（阴影、反光、纸张纹理）和装饰性元素（边框、校徽、标题、指北针）
- 完整保留所有墙体结构（外墙、内墙、隔断），线条连续无断裂
- 门的位置留空（不画门的弧线或任何门的标记），只保留墙体开口
- 保留房间编号、设施名称标注、图例区域
- 如果原图中存在"当前位置"标记，在原位置用红色实心圆点标出，并在图例中说明
- 不得添加、删除或移动任何结构元素，不得改变拓扑关系

只输出图像。`

    const rawOutput = await callImageModel(redrawPrompt, imageDataUrl, '3:4')
    const redrawnImageDataUrl = rawOutput

    await markSucceeded(generationId)
    await db.execute({
      sql: 'UPDATE tasks SET status = \'completed\', result_image_url = ?, finished_at = ? WHERE id = ?',
      args: [redrawnImageDataUrl, nowUnix(), taskId],
    })
  } catch (error: any) {
    const errorMsg = error?.message || 'unknown'
    console.error('Floor plan redraw failed:', errorMsg)
    await markFailedAndRefund(apiKeyId, generationId, errorMsg)
    await db.execute({
      sql: 'UPDATE tasks SET status = \'failed\', error_message = ?, finished_at = ? WHERE id = ?',
      args: [errorMsg.slice(0, 1000), nowUnix(), taskId],
    })
  }
}

app.get('/api/task/:id', async (c) => {
  const taskId = c.req.param('id')
  const apiKey = c.get('apiKey') as ApiKeyContext

  const result = await db.execute({
    sql: 'SELECT status, result_image_url, error_message FROM tasks WHERE id = ? AND api_key_id = ?',
    args: [taskId, apiKey.id],
  })

  const row = result.rows[0]
  if (!row) {
    return jsonError(c, 404, 'task_not_found', 'Task not found.')
  }

  const status = row.status as string

  if (status === 'completed') {
    return c.json({
      status: 'completed',
      wallMaskDataUrl: row.result_image_url as string,
      redrawnImageDataUrl: row.result_image_url as string,
      message: 'Task completed.',
    })
  }

  if (status === 'failed') {
    return c.json({
      status: 'failed',
      message: `Generation failed: ${row.error_message || 'unknown'}`,
    })
  }

  return c.json({ status: 'processing', message: 'Generating wall mask...' })
})

// --- Sync endpoints (start/end) detection ---

app.post('/api/endpoints', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }
  if (typeof body.destination !== 'string' || body.destination.trim() === '') {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing destination.')
  }

  const destination = body.destination.trim()

  return chargeAndRun(c, '/api/endpoints', async () => {
    // 使用 Qwen3-VL 原生 grounding 格式（point_2d，坐标范围 0~1000）
    const prompt = `Locate the following two points in this indoor floor plan image. Report point coordinates in JSON format.

1. "current_position": Find the "You Are Here" / "当前位置" / "Current Location" marker placed WITHIN THE MAP LAYOUT (among the rooms and corridors).

How to find it:
- Look at the legend/key area first to learn what the "当前位置" icon looks like (its shape and color). The legend usually shows a small sample icon next to the text "当前位置" or "Current Position" or "Entrance".
- Then scan the MAIN MAP AREA (the area with room outlines and corridors) for that SAME icon. It will be placed somewhere on or near a corridor to indicate where the viewer is standing.
- The marker could be: a triangle (▲, possibly rotated at any angle), a colored dot/circle, a star ★, an arrow ➤, a person icon, a pin, or any other distinctive symbol that differs from room labels.
- The marker may be small and subtle — look carefully along corridors and near entrances.
- Return the coordinates of this marker IN THE MAP AREA. NEVER return the coordinates of the legend/key area itself.

2. "destination_door_${destination}": The door/entrance of the room or facility labeled "${destination}" (or the closest match). The point should be at the doorway closest to the main corridor, NOT the center of the room.

Output format: JSON array with point_2d coordinates (0-1000 range) and labels. Example:
[{"point_2d": [420, 310], "label": "current_position"}, {"point_2d": [780, 550], "label": "destination_door"}]

If you cannot reliably locate a point, still include it but add "confidence": 0. Otherwise add "confidence" between 0.5 and 1.0 reflecting your certainty.`

    const rawText = await callVisionRaw(prompt, body.imageDataUrl)
    const parsed = parseGroundingResponse(rawText)

    return {
      start: parsed.start,
      end: parsed.end,
      message: parsed.message,
    }
  })
})

/**
 * 调用视觉模型并返回原始文本（不做 JSON parse）。
 * 用于 grounding 格式，模型可能返回 markdown code block 包裹的 JSON。
 */
async function callVisionRaw(prompt: string, imageDataUrl: string): Promise<string> {
  const response = await fetch(
    `${VISION_MODEL_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${VISION_MODEL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(60_000),
      body: JSON.stringify({
        model: VISION_MODEL_NAME,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageDataUrl } },
            ],
          },
        ],
        max_tokens: 1000,
        temperature: 0.1,
      }),
    }
  )

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Vision model failed: ${response.status} ${text.slice(0, 200)}`)
  }

  const data: any = await response.json()
  const content = data?.choices?.[0]?.message?.content
  if (!content || typeof content !== 'string') {
    throw new Error('Vision model returned empty response')
  }
  return content
}

type EndpointResult = {
  start: { x: number; y: number; confidence: number }
  end: { x: number; y: number; confidence: number }
  message: string
}

/**
 * 解析 Qwen3-VL grounding 格式的响应。
 * 模型输出 JSON 数组：[{"point_2d": [x, y], "label": "...", "confidence": 0.9}, ...]
 * 坐标范围 0~1000，需要除以 1000 转为归一化 [0, 1]。
 */
function parseGroundingResponse(rawText: string): EndpointResult {
  // 提取 JSON（可能被 markdown code block 包裹）
  let jsonStr = rawText.trim()
  const codeBlockMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/)
  if (codeBlockMatch) {
    jsonStr = codeBlockMatch[1].trim()
  }

  let points: any[]
  try {
    const parsed = JSON.parse(jsonStr)
    points = Array.isArray(parsed) ? parsed : []
  } catch {
    // 尝试从文本中提取 JSON 数组
    const arrayMatch = rawText.match(/\[[\s\S]*\]/)
    if (arrayMatch) {
      try {
        points = JSON.parse(arrayMatch[0])
      } catch {
        return fallbackResult('无法解析模型返回的坐标数据')
      }
    } else {
      return fallbackResult('模型未返回有效的坐标数据')
    }
  }

  if (!Array.isArray(points) || points.length === 0) {
    return fallbackResult('模型未返回有效的坐标点')
  }

  // 找 start（current_position）和 end（destination_door）
  let startItem: any = null
  let endItem: any = null

  for (const item of points) {
    if (!item || !Array.isArray(item.point_2d) || item.point_2d.length < 2) continue
    const label = String(item.label || '').toLowerCase()
    if (label.includes('current') || label.includes('start') || label.includes('位置')) {
      startItem = item
    } else if (label.includes('destination') || label.includes('door') || label.includes('end')) {
      endItem = item
    }
  }

  // 如果只有两个点且没匹配到 label，按顺序取
  if (!startItem && !endItem && points.length >= 2) {
    if (points[0]?.point_2d && points[1]?.point_2d) {
      startItem = points[0]
      endItem = points[1]
    }
  }

  const start = pointFromGrounding(startItem)
  const end = pointFromGrounding(endItem)

  const lowConfidence = start.confidence < 0.3 || end.confidence < 0.3
  const message = lowConfidence
    ? '部分坐标置信度较低，路径可能不准确。'
    : '已定位起点和终点。'

  return { start, end, message }
}

function pointFromGrounding(item: any): { x: number; y: number; confidence: number } {
  if (!item || !Array.isArray(item.point_2d) || item.point_2d.length < 2) {
    return { x: 0, y: 0, confidence: 0 }
  }

  const rawX = Number(item.point_2d[0])
  const rawY = Number(item.point_2d[1])

  if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) {
    return { x: 0, y: 0, confidence: 0 }
  }

  // Qwen3-VL grounding 坐标范围 0~1000，转为归一化 [0, 1]
  const x = Math.min(1, Math.max(0, rawX / 1000))
  const y = Math.min(1, Math.max(0, rawY / 1000))

  const confidence = typeof item.confidence === 'number' && Number.isFinite(item.confidence)
    ? Math.min(1, Math.max(0, item.confidence))
    : 0.7 // grounding 模式下模型不一定输出 confidence，默认给 0.7

  return { x, y, confidence }
}

function fallbackResult(message: string): EndpointResult {
  return {
    start: { x: 0, y: 0, confidence: 0 },
    end: { x: 0, y: 0, confidence: 0 },
    message,
  }
}

// --- Start server ---

let cvInitialized = false

async function initOpenCV(): Promise<void> {
  if (cvInitialized) return
  const readyPromise = new Promise<void>((resolve, reject) => {
    // Handle both: cv is a Promise (some builds) or cv has onRuntimeInitialized callback
    const cvModule = cv as unknown as { onRuntimeInitialized?: () => void; Mat?: unknown } | Promise<unknown>
    if (cvModule instanceof Promise) {
      cvModule.then(() => resolve()).catch(reject)
    } else if ((cvModule as { Mat?: unknown }).Mat) {
      // Already initialized
      resolve()
    } else {
      (cvModule as { onRuntimeInitialized: () => void }).onRuntimeInitialized = resolve
    }
  })
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('OpenCV WASM init timeout after 10s')), 10_000)
  )
  try {
    await Promise.race([readyPromise, timeout])
    cvInitialized = true
    console.log('OpenCV WASM initialized')
  } catch (err) {
    console.error('OpenCV init failed:', err)
    process.exit(1)
  }
}

await initDb()
await initOpenCV()
console.log(`FloorRoute API starting on port ${PORT}`)
const server = serve({ fetch: app.fetch, port: PORT })
;(server as any).requestTimeout = 600_000
;(server as any).headersTimeout = 600_000
;(server as any).keepAliveTimeout = 600_000
server.on('connection', (socket: import('net').Socket) => {
  socket.setKeepAlive(true, 30_000)
})
