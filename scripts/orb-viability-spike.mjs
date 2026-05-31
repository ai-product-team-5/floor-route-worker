/**
 * ORB viability spike — verifies that OpenCV ORB finds enough keypoints
 * on real wall mask images before committing to the full homography implementation.
 *
 * Usage: node scripts/orb-viability-spike.mjs
 * Exit 0 = PASS (all samples ≥ 30 keypoints), Exit 1 = FAIL or error
 */

import { createRequire } from 'module'
import { fileURLToPath } from 'url'
import path from 'path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Dynamic import to handle ESM/CJS quirks with @techstark/opencv-js
const { default: sharp } = await import('sharp')

const SAMPLE_IMAGES = [
  'C:\\Users\\xincy\\Documents\\Projects\\floor-route\\debug-output\\02_wall_mask.png',
  'C:\\Users\\xincy\\Documents\\Projects\\floor-route\\debug-output\\03_morph_closed.png',
  'C:\\Users\\xincy\\Documents\\Projects\\floor-route\\debug-output\\04_with_endpoints.png',
]

const MIN_KEYPOINTS = 30

// --- OpenCV init (handles both sync and async module patterns) ---
let cvModule
try {
  const imported = await import('@techstark/opencv-js')
  cvModule = imported.default ?? imported
} catch (err) {
  console.error('Failed to import @techstark/opencv-js:', err.message)
  process.exit(1)
}

// If it's a Promise, await it
if (cvModule instanceof Promise) {
  cvModule = await cvModule
}

// If onRuntimeInitialized hasn't fired yet, wait for it
if (!cvModule.Mat) {
  await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('OpenCV init timeout after 10s')), 10_000)
    cvModule.onRuntimeInitialized = () => {
      clearTimeout(t)
      resolve()
    }
  })
}

// --- Process each sample image ---
const counts = []

for (let i = 0; i < SAMPLE_IMAGES.length; i++) {
  const imagePath = SAMPLE_IMAGES[i]

  // Load image as raw RGBA buffer via sharp
  const { data, info } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  // Build OpenCV Mat from raw RGBA data
  const srcMat = cvModule.matFromArray(info.height, info.width, cvModule.CV_8UC4, data)

  // Convert to grayscale
  const grayMat = new cvModule.Mat()
  cvModule.cvtColor(srcMat, grayMat, cvModule.COLOR_RGBA2GRAY)

  // Run ORB
  const orb = new cvModule.ORB(1000)
  const kp = new cvModule.KeyPointVector()
  const desc = new cvModule.Mat()
  const emptyMask = new cvModule.Mat()

  orb.detectAndCompute(grayMat, emptyMask, kp, desc)

  const count = kp.size()
  counts.push(count)

  // Clean up all cv objects
  srcMat.delete()
  grayMat.delete()
  kp.delete()
  desc.delete()
  orb.delete()
  emptyMask.delete()
}

// --- Summary ---
const pass = counts.every(c => c >= MIN_KEYPOINTS)
const label = pass ? 'PASS' : 'FAIL'

const summaryLine = `ORB spike: sample1=${counts[0]} sample2=${counts[1]} sample3=${counts[2]} — ${label}`
console.log(summaryLine)

process.exit(pass ? 0 : 1)
