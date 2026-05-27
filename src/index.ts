import { Hono } from 'hono'
import { cors } from 'hono/cors'

type Env = {
  DB: D1Database

  VISION_MODEL_BASE_URL: string
  VISION_MODEL_API_KEY: string
  VISION_MODEL_NAME: string

  IMAGE_MODEL_BASE_URL: string
  IMAGE_MODEL_API_KEY: string
  IMAGE_MODEL_NAME: string
}

type ApiKeyContext = {
  id: string
  status: 'active' | 'disabled'
}

type Variables = {
  apiKey: ApiKeyContext
}

type Corner = {
  x: number
  y: number
}

type DestinationCandidate = {
  id: string
  title: string
  subtitle?: string
  confidence: number
}

const app = new Hono<{ Bindings: Env; Variables: Variables }>()

app.use(
  '/api/*',
  cors({
    origin: '*',
    allowMethods: ['GET', 'POST', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
  })
)

function jsonError(c: any, status: number, error: string, message: string) {
  return c.json({ error, message }, status)
}

function nowUnix() {
  return Math.floor(Date.now() / 1000)
}

function newId(prefix: string) {
  return `${prefix}_${crypto.randomUUID()}`
}

function getBearerToken(header: string | undefined): string | null {
  if (!header) return null

  const parts = header.split(' ')
  if (parts.length !== 2) return null
  if (parts[0] !== 'Bearer') return null
  if (!parts[1]) return null

  return parts[1]
}

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input)
  const hash = await crypto.subtle.digest('SHA-256', data)

  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

async function authMiddleware(c: any, next: any) {
  const token = getBearerToken(c.req.header('Authorization'))

  if (!token) {
    return jsonError(c, 401, 'missing_api_key', '請先設定 API Key。')
  }

  const keyHash = await sha256Hex(token)

  const row = await c.env.DB.prepare(
    `
    SELECT id, status
    FROM api_keys
    WHERE key_hash = ?
    LIMIT 1
    `
  )
    .bind(keyHash)
    .first<ApiKeyContext>()

  if (!row) {
    return jsonError(c, 401, 'invalid_api_key', 'API Key 無效。')
  }

  if (row.status !== 'active') {
    return jsonError(c, 403, 'disabled_api_key', 'API Key 已被禁用。')
  }

  c.set('apiKey', {
    id: row.id,
    status: row.status,
  })

  await next()
}

app.use('/api/*', authMiddleware)

async function readJsonBody(c: any): Promise<any | null> {
  try {
    return await c.req.json()
  } catch {
    return null
  }
}

function isDataUrl(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^data:image\/(png|jpeg|jpg|webp);base64,/.test(value)
  )
}

function validateCorners(corners: any): corners is Corner[] {
  if (!Array.isArray(corners)) return false
  if (corners.length !== 4) return false

  return corners.every((p) => {
    return (
      p &&
      typeof p.x === 'number' &&
      typeof p.y === 'number' &&
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      p.x >= 0 &&
      p.x <= 1 &&
      p.y >= 0 &&
      p.y <= 1
    )
  })
}

function fallbackCorners(): Corner[] {
  return [
    { x: 0.05, y: 0.05 },
    { x: 0.95, y: 0.05 },
    { x: 0.95, y: 0.95 },
    { x: 0.05, y: 0.95 },
  ]
}

function normalizeLimit(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 5
  return Math.min(10, Math.max(1, Math.floor(value)))
}

function validateCandidates(value: any, limit: number): DestinationCandidate[] {
  if (!Array.isArray(value)) return []

  return value
    .filter((item) => {
      return (
        item &&
        typeof item.id === 'string' &&
        typeof item.title === 'string' &&
        typeof item.confidence === 'number' &&
        item.confidence >= 0 &&
        item.confidence <= 1
      )
    })
    .slice(0, limit)
    .map((item) => ({
      id: item.id,
      title: item.title,
      subtitle: typeof item.subtitle === 'string' ? item.subtitle : undefined,
      confidence: item.confidence,
    }))
}

async function callVisionJson(
  env: Env,
  prompt: string,
  imageDataUrl: string
): Promise<any> {
  const response = await fetch(
    `${env.VISION_MODEL_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.VISION_MODEL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: env.VISION_MODEL_NAME,
        temperature: 0,
        max_tokens: 1000,
        response_format: { type: 'json_object' },
        messages: [
          {
            role: 'system',
            content:
              '你是一個嚴格的 JSON API。你只能輸出合法 JSON，不要輸出 Markdown，不要輸出解釋。',
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: prompt,
              },
              {
                type: 'image_url',
                image_url: {
                  url: imageDataUrl,
                },
              },
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

function dataUrlToFile(dataUrl: string, filename: string): File {
  const match = dataUrl.match(/^data:(image\/(?:png|jpeg|jpg|webp));base64,(.+)$/)

  if (!match) {
    throw new Error('Invalid image data URL')
  }

  const mime = match[1] === 'image/jpg' ? 'image/jpeg' : match[1]
  const base64 = match[2]
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }

  return new File([bytes], filename, { type: mime })
}

async function callImageEdit(
  env: Env,
  imageDataUrl: string,
  destination: string
): Promise<string> {
  const file = dataUrlToFile(imageDataUrl, 'floor-plan.png')

  const prompt = `
這是一張室內平面圖。
請在圖上用清晰的紅色虛線標註從当前位置（入口附近）到「${destination}」的最佳步行路線。
路線必須沿走廊和通道行走，不能穿牆。
在起點標註綠色圓點，終點標註紅色圓點。
保持原始平面圖清晰可見。
`

  const form = new FormData()
  form.append('model', env.IMAGE_MODEL_NAME)
  form.append('image[]', file)
  form.append('prompt', prompt)

  const response = await fetch(
    `${env.IMAGE_MODEL_BASE_URL.replace(/\/+$/, '')}/images/edits`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.IMAGE_MODEL_API_KEY}`,
      },
      body: form,
    }
  )

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Image model failed: ${response.status} ${text}`)
  }

  const data: any = await response.json()
  const b64 = data.data?.[0]?.b64_json

  if (typeof b64 !== 'string') {
    throw new Error('Image model did not return b64_json')
  }

  return `data:image/png;base64,${b64}`
}

async function createGeneration(
  env: Env,
  apiKeyId: string,
  endpoint: string
): Promise<string> {
  const generationId = newId('gen')
  const now = nowUnix()

  await env.DB.prepare(
    `
    INSERT INTO generations
      (id, api_key_id, endpoint, cost_credits, status, created_at)
    VALUES
      (?, ?, ?, 1, 'pending', ?)
    `
  )
    .bind(generationId, apiKeyId, endpoint, now)
    .run()

  return generationId
}

async function deductOneCredit(
  env: Env,
  apiKeyId: string,
  generationId: string
): Promise<boolean> {
  const now = nowUnix()

  const updateResult = await env.DB.prepare(
    `
    UPDATE api_key_credit_accounts
    SET balance = balance - 1, updated_at = ?
    WHERE api_key_id = ? AND balance >= 1
    `
  )
    .bind(now, apiKeyId)
    .run()

  if (updateResult.meta.changes !== 1) {
    return false
  }

  await env.DB.prepare(
    `
    INSERT INTO api_key_credit_ledger
      (id, api_key_id, type, amount, ref_id, created_at)
    VALUES
      (?, ?, 'consume', -1, ?, ?)
    `
  )
    .bind(newId('ledger'), apiKeyId, generationId, now)
    .run()

  return true
}

async function markGenerationSucceeded(
  env: Env,
  generationId: string
): Promise<void> {
  await env.DB.prepare(
    `
    UPDATE generations
    SET status = 'succeeded', finished_at = ?
    WHERE id = ?
    `
  )
    .bind(nowUnix(), generationId)
    .run()
}

async function markGenerationFailedAndRefund(
  env: Env,
  apiKeyId: string,
  generationId: string,
  errorMessage: string
): Promise<void> {
  const now = nowUnix()

  await env.DB.batch([
    env.DB.prepare(
      `
      UPDATE generations
      SET status = 'failed', error_message = ?, finished_at = ?
      WHERE id = ?
      `
    ).bind(errorMessage.slice(0, 1000), now, generationId),

    env.DB.prepare(
      `
      UPDATE api_key_credit_accounts
      SET balance = balance + 1, updated_at = ?
      WHERE api_key_id = ?
      `
    ).bind(now, apiKeyId),

    env.DB.prepare(
      `
      INSERT INTO api_key_credit_ledger
        (id, api_key_id, type, amount, ref_id, created_at)
      VALUES
        (?, ?, 'refund', 1, ?, ?)
      `
    ).bind(newId('ledger'), apiKeyId, generationId, now),
  ])
}

async function markGenerationFailedNoRefund(
  env: Env,
  generationId: string,
  errorMessage: string
): Promise<void> {
  await env.DB.prepare(
    `
    UPDATE generations
    SET status = 'failed', error_message = ?, finished_at = ?
    WHERE id = ?
    `
  )
    .bind(errorMessage.slice(0, 1000), nowUnix(), generationId)
    .run()
}

async function chargeAndRun(c: any, endpoint: string, fn: () => Promise<any>) {
  const apiKey = c.get('apiKey') as ApiKeyContext
  const generationId = await createGeneration(c.env, apiKey.id, endpoint)

  const deducted = await deductOneCredit(c.env, apiKey.id, generationId)

  if (!deducted) {
    await markGenerationFailedNoRefund(
      c.env,
      generationId,
      'insufficient_credits'
    )

    return jsonError(c, 402, 'insufficient_credits', '額度不足。')
  }

  try {
    const result = await fn()
    await markGenerationSucceeded(c.env, generationId)
    return c.json(result)
  } catch (error: any) {
    console.error('Model request failed:', error?.message || error)

    await markGenerationFailedAndRefund(
      c.env,
      apiKey.id,
      generationId,
      error?.message || 'model_request_failed'
    )

    return jsonError(
      c,
      500,
      'model_request_failed',
      'AI 模型調用失敗，請稍後重試。'
    )
  }
}

app.get('/', (c) => {
  return c.json({
    message: 'FloorRoute API is running.',
    routes: [
      'GET /api/credits',
      'POST /api/corner',
      'POST /api/search',
      'POST /api/path',
    ],
  })
})

app.get('/api/credits', async (c) => {
  const apiKey = c.get('apiKey') as ApiKeyContext

  const row = await c.env.DB.prepare(
    `
    SELECT balance
    FROM api_key_credit_accounts
    WHERE api_key_id = ?
    LIMIT 1
    `
  )
    .bind(apiKey.id)
    .first<{ balance: number }>()

  return c.json({
    balance: row?.balance ?? 0,
  })
})

app.post('/api/corner', async (c) => {
  const body = await readJsonBody(c)

  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(
      c,
      400,
      'invalid_request',
      '請求格式錯誤：缺少 imageDataUrl。'
    )
  }

  return chargeAndRun(c, '/api/corner', async () => {
    const result = await callVisionJson(
      c.env,
      `
你是一个文档边角检测器。请找到图片中矩形标牌/平面图的四个角。
返回归一化坐标（0到1），顺序为左上、右上、右下、左下。
只输出 JSON：
{
  "corners": [
    {"x": 0.12, "y": 0.05},
    {"x": 0.88, "y": 0.06},
    {"x": 0.87, "y": 0.92},
    {"x": 0.11, "y": 0.91}
  ],
  "message": "已识别平面图边框。"
}
如果无法可靠识别，请仍然返回最接近的四个角。
`,
      body.imageDataUrl
    )

    if (!validateCorners(result.corners)) {
      return {
        corners: fallbackCorners(),
        message: '未能準確識別平面圖邊框，請手動調整四角。',
      }
    }

    return {
      corners: result.corners,
      message:
        typeof result.message === 'string'
          ? result.message
          : '已識別平面圖邊框。',
    }
  })
})

app.post('/api/search', async (c) => {
  const body = await readJsonBody(c)

  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(
      c,
      400,
      'invalid_request',
      '請求格式錯誤：缺少 imageDataUrl。'
    )
  }

  if (typeof body.query !== 'string' || body.query.trim() === '') {
    return jsonError(
      c,
      400,
      'invalid_request',
      '請求格式錯誤：缺少 query。'
    )
  }

  const query = body.query.trim()
  const limit = normalizeLimit(body.limit)

  return chargeAndRun(c, '/api/search', async () => {
    const result = await callVisionJson(
      c.env,
      `
你是室內平面圖目的地搜尋器。
用戶給你一張已校正的室內平面圖和搜索詞：「${query}」。

要求：
1. 只根據圖中可見的文字、房間、設施返回匹配候選。
2. 不要編造圖中看不到的地點。
3. 最多返回 ${limit} 個。
4. candidates 按 confidence 由高到低排序。
5. 只輸出 JSON。

JSON 格式：
{
  "message": "已找到可能匹配的目的地，請選擇最準確的一項。",
  "candidates": [
    {
      "id": "restroom-east",
      "title": "衛生間",
      "subtitle": "東側電梯廳旁",
      "confidence": 0.92
    }
  ]
}
如果沒有匹配，返回：
{
  "message": "未找到匹配的目的地。",
  "candidates": []
}
`,
      body.imageDataUrl
    )

    const candidates = validateCandidates(result.candidates, limit)

    return {
      candidates,
      message:
        typeof result.message === 'string'
          ? result.message
          : candidates.length > 0
            ? '已找到可能匹配的目的地，請選擇最準確的一項。'
            : '未找到匹配的目的地。',
    }
  })
})

app.post('/api/path', async (c) => {
  const body = await readJsonBody(c)

  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(
      c,
      400,
      'invalid_request',
      '請求格式錯誤：缺少 imageDataUrl。'
    )
  }

  if (typeof body.destination !== 'string' || body.destination.trim() === '') {
    return jsonError(
      c,
      400,
      'invalid_request',
      '請求格式錯誤：缺少 destination。'
    )
  }

  const destination = body.destination.trim()

  return chargeAndRun(c, '/api/path', async () => {
    const resultImageUrl = await callImageEdit(
      c.env,
      body.imageDataUrl,
      destination
    )

    return {
      resultImageUrl,
      message: '已生成導航路線。',
    }
  })
})

export default app