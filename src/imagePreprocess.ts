import sharp from 'sharp'
import { imageSize } from 'image-size'

export type PadInfo = {
  padLeft: number
  padTop: number
  paddedW: number
  paddedH: number
  originalW: number
  originalH: number
  aspectRatio: string
}

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

/**
 * Preprocess an image for the image generation model:
 * - Determine orientation (landscape → 2048x1536, portrait → 1536x2048)
 * - Scale to fit inside target canvas preserving aspect ratio
 * - Pad with white to exact target dimensions (centered)
 */
export async function preprocessImage(imageDataUrl: string): Promise<{ paddedDataUrl: string; padInfo: PadInfo }> {
  const dimensions = parseImageDimensions(imageDataUrl)
  if (!dimensions) {
    throw new Error('无法解析输入图片尺寸')
  }

  const { width, height } = dimensions

  // Determine target canvas based on orientation
  const isLandscape = width > height
  const targetW = isLandscape ? 2048 : 1536
  const targetH = isLandscape ? 1536 : 2048
  const aspectRatio = isLandscape ? '4:3' : '3:4'

  // Scale image to fit inside target canvas
  const scale = Math.min(targetW / width, targetH / height)
  const scaledW = Math.round(width * scale)
  const scaledH = Math.round(height * scale)

  // Calculate padding to center the image
  const padLeft = Math.round((targetW - scaledW) / 2)
  const padTop = Math.round((targetH - scaledH) / 2)

  const base64Match = imageDataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
  if (!base64Match) {
    throw new Error('Invalid data URL format')
  }

  const inputBuf = Buffer.from(base64Match[1], 'base64')

  const paddedBuf = await sharp(inputBuf)
    .resize(scaledW, scaledH, { fit: 'fill' })
    .extend({
      top: padTop,
      bottom: targetH - scaledH - padTop,
      left: padLeft,
      right: targetW - scaledW - padLeft,
      background: { r: 255, g: 255, b: 255, alpha: 1 },
    })
    .png()
    .toBuffer()

  const paddedDataUrl = `data:image/png;base64,${paddedBuf.toString('base64')}`

  const padInfo: PadInfo = {
    padLeft,
    padTop,
    paddedW: targetW,
    paddedH: targetH,
    originalW: scaledW,
    originalH: scaledH,
    aspectRatio,
  }

  console.log(`Preprocessed: ${width}x${height} → scaled ${scaledW}x${scaledH} → padded ${targetW}x${targetH} (${aspectRatio})`)

  return { paddedDataUrl, padInfo }
}

/**
 * Crop padding from an output image using the padInfo from preprocessing.
 * The output image may be a different resolution than the padded input,
 * so we scale the crop region accordingly.
 */
export async function cropPadding(outputDataUrl: string, padInfo: PadInfo): Promise<string> {
  const dimensions = parseImageDimensions(outputDataUrl)
  if (!dimensions) {
    throw new Error('无法解析输出图片尺寸')
  }

  const { width: outputW, height: outputH } = dimensions

  // Scale factors from padded canvas to output resolution
  const scaleX = outputW / padInfo.paddedW
  const scaleY = outputH / padInfo.paddedH

  // Calculate crop region in output coordinates
  const left = Math.round(padInfo.padLeft * scaleX)
  const top = Math.round(padInfo.padTop * scaleY)
  const cropW = Math.round(padInfo.originalW * scaleX)
  const cropH = Math.round(padInfo.originalH * scaleY)

  const base64Match = outputDataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
  if (!base64Match) {
    throw new Error('Invalid data URL format')
  }

  const inputBuf = Buffer.from(base64Match[1], 'base64')

  const croppedBuf = await sharp(inputBuf)
    .extract({ left, top, width: cropW, height: cropH })
    .png()
    .toBuffer()

  return `data:image/png;base64,${croppedBuf.toString('base64')}`
}
