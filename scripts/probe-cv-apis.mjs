import cvLib from '@techstark/opencv-js'

const cv = await (cvLib instanceof Promise ? cvLib : new Promise((r, rej) => {
  const t = setTimeout(() => rej(new Error('cv init timeout')), 30000)
  if (cvLib.Mat) { clearTimeout(t); r(cvLib) }
  else cvLib.onRuntimeInitialized = () => { clearTimeout(t); r(cvLib) }
}))

const apis = [
  'phaseCorrelate', 'findTransformECC',
  'LineSegmentDetector', 'createLineSegmentDetector',
  'HoughLinesP', 'HoughLines',
  'distanceTransform', 'matchTemplate',
  'Canny', 'Sobel', 'Scharr',
  'findHomography', 'warpPerspective',
  'estimateAffinePartial2D', 'estimateAffine2D',
  'MOTION_TRANSLATION', 'MOTION_EUCLIDEAN', 'MOTION_AFFINE', 'MOTION_HOMOGRAPHY',
  'TermCriteria', 'TERM_CRITERIA_EPS', 'TERM_CRITERIA_COUNT',
  'goodFeaturesToTrack', 'calcOpticalFlowPyrLK',
  'createHanningWindow',
  'morphologyEx', 'MORPH_GRADIENT',
]
for (const a of apis) {
  const v = cv[a]
  console.log(a.padEnd(32), v === undefined ? 'NO' : typeof v === 'function' ? 'fn' : typeof v === 'number' ? `num=${v}` : typeof v)
}
