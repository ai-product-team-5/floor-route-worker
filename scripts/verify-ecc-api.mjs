import cvLib from '@techstark/opencv-js'

// Init: @techstark/opencv-js@4.12.0-release.1 in Node ESM exports an Emscripten
// factory whose `then` is single-arg (does NOT honor onReject). Three pitfalls:
//   1. `await cvLib` HANGS — Node calls then(ok, err) with 2 args; thenable ignores err.
//   2. `resolve(cvLib)` from new Promise HANGS — same Promise assimilation issue.
//   3. Returning `cvLib` from a `.then(() => cvLib)` handler HANGS — same.
// Workaround: register a single-arg .then(cb), resolve with a sentinel, and
// access cvLib as a free variable AFTER the await — never let Promises see cvLib.
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error('init timeout 60s')), 60_000)
  if (cvLib.Mat) { clearTimeout(timer); resolve(true); return }
  if (typeof cvLib.then === 'function') {
    cvLib.then(() => { clearTimeout(timer); resolve(true) })
  } else {
    cvLib.onRuntimeInitialized = () => { clearTimeout(timer); resolve(true) }
  }
})
const cv = cvLib

console.log('[init] OK, cv.Mat present:', !!cv.Mat)

// 1. typeof / constants
// NOTE: in this WASM build, TermCriteria flags are exposed as TermCriteria_EPS / TermCriteria_COUNT
// (NOT TERM_CRITERIA_EPS as the d.ts claims). Values: TermCriteria_EPS=2, TermCriteria_COUNT=1.
const TC_EPS = cv.TermCriteria_EPS ?? cv.TERM_CRITERIA_EPS
const TC_COUNT = cv.TermCriteria_COUNT ?? cv.TERM_CRITERIA_COUNT

console.log('findTransformECC:', typeof cv.findTransformECC)
console.log('MOTION_HOMOGRAPHY:', cv.MOTION_HOMOGRAPHY)
console.log('MOTION_AFFINE:', cv.MOTION_AFFINE)
console.log('MOTION_EUCLIDEAN:', cv.MOTION_EUCLIDEAN)
console.log('MOTION_TRANSLATION:', cv.MOTION_TRANSLATION)
console.log('TermCriteria_EPS:', cv.TermCriteria_EPS, '(d.ts name TERM_CRITERIA_EPS:', cv.TERM_CRITERIA_EPS, ')')
console.log('TermCriteria_COUNT:', cv.TermCriteria_COUNT, '(d.ts name TERM_CRITERIA_COUNT:', cv.TERM_CRITERIA_COUNT, ')')

const W = 100, H = 100

// 2. case 1: identity (template === input) — ECC should be ~1.0
const tmpl = new cv.Mat(H, W, cv.CV_8UC1)
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) tmpl.ucharPtr(y, x)[0] = x * 2
const input = tmpl.clone()
const warpMatrix = cv.Mat.eye(3, 3, cv.CV_32F)
const criteria = new cv.TermCriteria(TC_EPS + TC_COUNT, 50, 1e-3)
// In this WASM build findTransformECC has NO default args — must pass all 7.
// inputMask = empty Mat means "no mask". gaussFiltSize = 5 is the OpenCV default.
const noMask = new cv.Mat()
const gaussFiltSize = 5

console.log('--- calling findTransformECC (identity case) ---')
let case1 = { ok: false }
try {
  const ecc = cv.findTransformECC(tmpl, input, warpMatrix, cv.MOTION_HOMOGRAPHY, criteria, noMask, gaussFiltSize)
  console.log('SUCCESS: ecc =', ecc)
  console.log('warpMatrix:')
  const rows = []
  for (let i = 0; i < 3; i++) {
    const row = []
    for (let j = 0; j < 3; j++) row.push(warpMatrix.floatAt(i, j).toFixed(4))
    rows.push(row)
    console.log(' ', row.join(' '))
  }
  case1 = { ok: true, ecc, rows }
} catch (e) {
  console.log('FAIL:', e?.message || e)
  case1 = { ok: false, err: String(e?.message || e) }
}

// 3. case 2: shifted (dx=2)
const shifted = new cv.Mat(H, W, cv.CV_8UC1)
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const src = Math.max(0, x - 2)
  shifted.ucharPtr(y, x)[0] = src * 2
}
const warp2 = cv.Mat.eye(3, 3, cv.CV_32F)
console.log('--- shift dx=2 case ---')
let case2 = { ok: false }
try {
  const ecc = cv.findTransformECC(tmpl, shifted, warp2, cv.MOTION_HOMOGRAPHY, criteria, noMask, gaussFiltSize)
  console.log('SUCCESS: ecc =', ecc)
  const tx = warp2.floatAt(0, 2)
  const ty = warp2.floatAt(1, 2)
  console.log('tx (should be ±2):', tx.toFixed(2))
  console.log('ty (should be 0):', ty.toFixed(2))
  case2 = { ok: true, ecc, tx, ty }
} catch (e) {
  console.log('FAIL:', e?.message || e)
  case2 = { ok: false, err: String(e?.message || e) }
}

console.log('--- summary ---')
console.log(JSON.stringify({ case1, case2 }, null, 2))

tmpl.delete(); input.delete(); shifted.delete()
warpMatrix.delete(); warp2.delete(); noMask.delete()

console.log('--- done ---')
process.exit(0)
