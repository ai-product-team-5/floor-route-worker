// ECC convergence probe — runs findTransformECC against a real photo+mask pair
// at multiple motion types and prints a structured GO/NO_GO verdict.
//
// Usage: node scripts/ecc-probe.mjs <photo_path> <mask_path>
//
// CRITICAL: @techstark/opencv-js exposes a thenable (Emscripten Module) that is
// the SAME object before and after init. If we do `resolve(readyCv)` from
// inside `cvLib.then(cb)`, the outer Promise assimilates the still-thenable cv
// via Promise.resolve(thenable).then(resolve, reject), which hangs because the
// loader internally calls reject(undefined) after init. We must capture cv
// through a closure variable and resolve() with NO argument.

import { promises as fs, appendFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import { performance } from 'node:perf_hooks'
import sharp from 'sharp'
import cvLib from '@techstark/opencv-js'

const args = process.argv.slice(2)
if (args.length < 2) {
  console.error('Usage: node scripts/ecc-probe.mjs <photo_path> <mask_path>')
  process.exit(2)
}
const [photoPath, maskPath] = args

// Per-stage progress goes both to stdout (unbuffered via process.stdout.write
// + sync file append) so we can see where the script lives if it hangs.
const progressLog = path.resolve(process.cwd(), 'debug-output', 'ecc-probe.progress.log')
mkdirSync(path.dirname(progressLog), { recursive: true })
function stage(label) {
  const line = `[${new Date().toISOString()}] ${label}\n`
  try { appendFileSync(progressLog, line) } catch { /* ignore */ }
  process.stdout.write(line)
}

stage('start')

// ---------------------------------------------------------------------------
// 1. Init OpenCV WASM via .then(cb), capturing cv through closure (NOT via
//    resolve(cv) — that would assimilate the still-thenable Module).
// ---------------------------------------------------------------------------
let cv
await new Promise((resolve, reject) => {
  const t = setTimeout(() => reject(new Error('cv init timeout 30s')), 30_000)
  cvLib.then((readyCv) => {
    clearTimeout(t)
    cv = readyCv
    resolve() // <- IMPORTANT: no argument, otherwise assimilation hangs
  })
})
stage(`cv-init done (Mat? ${typeof cv.Mat === 'function'})`)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const toDelete = []
function track(mat) { if (mat) toDelete.push(mat); return mat }

function fmt(n, digits = 4) {
  if (!Number.isFinite(n)) return String(n)
  return n.toFixed(digits)
}

function printMat(mat, rows, cols) {
  const out = []
  const data = mat.data32F
  for (let r = 0; r < rows; r++) {
    const row = []
    for (let c = 0; c < cols; c++) {
      row.push(fmt(data[r * cols + c], 6).padStart(12))
    }
    out.push('    ' + row.join(' '))
  }
  return out.join('\n')
}

async function loadAsRgbaMat(filePath, targetW, targetH) {
  let pipeline = sharp(filePath)
  if (targetW && targetH) pipeline = pipeline.resize(targetW, targetH, { fit: 'fill' })
  const { data, info } = await pipeline.ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const mat = cv.matFromArray(info.height, info.width, cv.CV_8UC4, data)
  return { mat: track(mat), width: info.width, height: info.height }
}

async function getOriginalSize(filePath) {
  const meta = await sharp(filePath).metadata()
  return { width: meta.width, height: meta.height }
}

async function saveCvFloatAsPng(mat, outPath) {
  // mat is CV_32F in [0,1]. Scale to 0-255 uint8 for visual inspection.
  const u8 = new cv.Mat()
  track(u8)
  mat.convertTo(u8, cv.CV_8U, 255.0, 0.0)
  const buf = Buffer.from(u8.data)
  await fs.mkdir(path.dirname(outPath), { recursive: true })
  await sharp(buf, { raw: { width: u8.cols, height: u8.rows, channels: 1 } })
    .png()
    .toFile(outPath)
}

// ---------------------------------------------------------------------------
// 2. Preprocess: photo + mask → distance transform (CV_32F in [0,1])
// ---------------------------------------------------------------------------
const preprocessStart = performance.now()

const origPhoto = await getOriginalSize(photoPath)
const origMask = await getOriginalSize(maskPath)
const workW = Math.min(800, origPhoto.width)
const workH = Math.round(origPhoto.height * (workW / origPhoto.width))
stage(`work resolution = ${workW}x${workH}`)

const photoRgba = await loadAsRgbaMat(photoPath, workW, workH)
const maskRgba = await loadAsRgbaMat(maskPath, workW, workH)
stage('rgba loaded')

// --- Photo pipeline
const photoGray = track(new cv.Mat())
cv.cvtColor(photoRgba.mat, photoGray, cv.COLOR_RGBA2GRAY)
stage('photo gray')

const photoCanny = track(new cv.Mat())
cv.Canny(photoGray, photoCanny, 50, 150)
stage('photo canny')

const photoCannyInv = track(new cv.Mat())
cv.bitwise_not(photoCanny, photoCannyInv)

const photoDt = track(new cv.Mat())
cv.distanceTransform(photoCannyInv, photoDt, cv.DIST_L2, 5)
stage('photo distanceTransform')

const photoDtNorm = track(new cv.Mat())
cv.normalize(photoDt, photoDtNorm, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
stage('photo normalize')

// --- Mask pipeline
const maskGray = track(new cv.Mat())
cv.cvtColor(maskRgba.mat, maskGray, cv.COLOR_RGBA2GRAY)
stage('mask gray')

const maskBin = track(new cv.Mat())
cv.threshold(maskGray, maskBin, 127, 255, cv.THRESH_BINARY_INV)

const maskDilated = track(new cv.Mat())
const kernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3)))
cv.dilate(maskBin, maskDilated, kernel, new cv.Point(-1, -1), 1)
stage('mask dilate')

const maskCanny = track(new cv.Mat())
cv.Canny(maskDilated, maskCanny, 50, 150)
stage('mask canny')

const maskCannyInv = track(new cv.Mat())
cv.bitwise_not(maskCanny, maskCannyInv)

const maskDt = track(new cv.Mat())
cv.distanceTransform(maskCannyInv, maskDt, cv.DIST_L2, 5)
stage('mask distanceTransform')

const maskDtNorm = track(new cv.Mat())
cv.normalize(maskDt, maskDtNorm, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
stage('mask normalize')

const preprocessMs = Math.round(performance.now() - preprocessStart)

// --- Save visualization PNGs
const outDir = path.resolve(process.cwd(), 'debug-output')
const photoDtOut = path.join(outDir, 'probe_photo_dt.png')
const maskDtOut = path.join(outDir, 'probe_mask_dt.png')
await saveCvFloatAsPng(photoDtNorm, photoDtOut)
await saveCvFloatAsPng(maskDtNorm, maskDtOut)
stage('dt pngs saved')

// ---------------------------------------------------------------------------
// 3. findTransformECC three times
// ---------------------------------------------------------------------------
const trials = [
  { name: 'HOMOGRAPHY', motionType: cv.MOTION_HOMOGRAPHY, matrixRows: 3, matrixCols: 3, maxIter: 100, eps: 1e-4 },
  { name: 'AFFINE',     motionType: cv.MOTION_AFFINE,     matrixRows: 2, matrixCols: 3, maxIter: 50,  eps: 1e-3 },
  { name: 'EUCLIDEAN',  motionType: cv.MOTION_EUCLIDEAN,  matrixRows: 2, matrixCols: 3, maxIter: 50,  eps: 1e-3 },
]

const results = []
for (const t of trials) {
  stage(`trial start ${t.name}`)
  const trialDelete = []
  const trackTrial = (m) => { if (m) trialDelete.push(m); return m }
  let status = 'success'
  let ecc = null
  let matrixDump = ''
  let errMessage = null
  const t0 = performance.now()
  try {
    const warpMatrix = trackTrial(cv.Mat.eye(t.matrixRows, t.matrixCols, cv.CV_32F))
    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_COUNT,
      t.maxIter,
      t.eps,
    )
    const inputMask = trackTrial(new cv.Mat()) // empty = no mask
    ecc = cv.findTransformECC(
      maskDtNorm,
      photoDtNorm,
      warpMatrix,
      t.motionType,
      criteria,
      inputMask,
      5,
    )
    matrixDump = printMat(warpMatrix, t.matrixRows, t.matrixCols)
  } catch (err) {
    let raw = err
    if (typeof err === 'number' && typeof cv.exceptionFromPtr === 'function') {
      try { raw = cv.exceptionFromPtr(err) } catch { /* ignore */ }
    }
    errMessage = (raw && raw.msg) || (raw && raw.message) || String(raw)
    status = /StsNoConv|did not converge|not converged|noConv/i.test(errMessage) ? 'noConv' : 'error'
  } finally {
    for (const m of trialDelete) {
      try { m.delete() } catch { /* ignore */ }
    }
  }
  const ms = Math.round(performance.now() - t0)
  stage(`trial end ${t.name} status=${status} ecc=${ecc} ms=${ms}`)
  results.push({ name: t.name, status, ecc, matrixDump, errMessage, ms })
}

// ---------------------------------------------------------------------------
// 4. Print summary + VERDICT
// ---------------------------------------------------------------------------
const out = []
out.push('=== ECC Probe Results ===')
out.push(`photo: ${photoPath} (${origPhoto.width}x${origPhoto.height})`)
out.push(`mask: ${maskPath} (${origMask.width}x${origMask.height})`)
out.push(`work resolution: ${workW}x${workH}`)
out.push(`preprocess time: ${preprocessMs}ms`)
out.push(`debug: ${photoDtOut}`)
out.push(`debug: ${maskDtOut}`)
out.push('')

for (const r of results) {
  const statusTag = r.status
  const eccStr = r.ecc !== null && Number.isFinite(r.ecc) ? fmt(r.ecc, 4) : 'N/A'
  let line = `[${r.name.padEnd(12)}] status=${statusTag}  ecc=${eccStr}  ms=${r.ms}`
  if (r.errMessage) line += `  err="${r.errMessage}"`
  out.push(line)
  if (r.matrixDump) {
    out.push('  matrix:')
    out.push(r.matrixDump)
  }
}
out.push('')

const successResults = results.filter((r) => r.status === 'success' && Number.isFinite(r.ecc))
let bestMotion = 'none'
let bestEcc = -Infinity
for (const r of successResults) {
  if (r.ecc > bestEcc) {
    bestEcc = r.ecc
    bestMotion = r.name.toLowerCase()
  }
}
const verdict = bestEcc >= 0.3 ? 'GO' : 'NO_GO'

out.push(`VERDICT: ${verdict}`)
out.push(`  best motion type: ${bestMotion}`)
out.push(`  best ecc: ${bestEcc === -Infinity ? 'N/A' : fmt(bestEcc, 4)}`)
out.push('  GO threshold: best ecc >= 0.3')

console.log(out.join('\n'))

// ---------------------------------------------------------------------------
// 5. Cleanup
// ---------------------------------------------------------------------------
for (const m of toDelete) {
  try { m.delete() } catch { /* ignore */ }
}

stage('done')
process.exit(0)