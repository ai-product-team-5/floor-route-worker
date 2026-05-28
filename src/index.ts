import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { createClient } from '@libsql/client'
import { createHash, randomUUID } from 'node:crypto'

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
  `)
}

// --- Types ---

type ApiKeyContext = {
  id: string
  status: 'active' | 'disabled'
}

type Corner = { x: number; y: number }

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

function validateCorners(corners: any): corners is Corner[] {
  if (!Array.isArray(corners) || corners.length !== 4) return false
  return corners.every(
    (p) =>
      p &&
      typeof p.x === 'number' &&
      typeof p.y === 'number' &&
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      p.x >= 0 && p.x <= 1 &&
      p.y >= 0 && p.y <= 1
  )
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
      body: JSON.stringify({
        model: VISION_MODEL_NAME,
        temperature: 0,
        max_tokens: 1000,
        response_format: { type: 'json_object' },
        messages: [
          {
            role: 'system',
            content: '你是一个严格的 JSON API。你只能输出合法 JSON，不要输出 Markdown，不要输出解释。',
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

async function callImageEdit(imageDataUrl: string, destination: string): Promise<string> {
  const prompt = `
这是一张室内平面图。
请在图上用清晰的红色虚线标注从当前位置（入口附近）到「${destination}」的最佳步行路线。
路线必须沿走廊和通道行走，不能穿墙。
在起点标注绿色圆点，终点标注红色圆点。
保持原始平面图清晰可见。
`

  const response = await fetch(
    `${IMAGE_MODEL_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${IMAGE_MODEL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: IMAGE_MODEL_NAME,
        modalities: ['image', 'text'],
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

  const imageUrl = message?.images?.[0]?.image_url?.url
  if (typeof imageUrl === 'string' && imageUrl.startsWith('data:')) {
    return imageUrl
  }

  if (Array.isArray(message?.content)) {
    for (const part of message.content) {
      if (part.type === 'image_url' && part.image_url?.url?.startsWith('data:')) {
        return part.image_url.url
      }
    }
  }

  throw new Error('Image model did not return a generated image')
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
  if (!token) return jsonError(c, 401, 'missing_api_key', '请先设置 API Key。')

  const keyHash = sha256Hex(token)
  const row = await getApiKey(keyHash)
  if (!row) return jsonError(c, 401, 'invalid_api_key', 'API Key 无效。')
  if (row.status !== 'active') return jsonError(c, 403, 'disabled_api_key', 'API Key 已被禁用。')

  c.set('apiKey', row)
  await next()
})

async function chargeAndRun(c: any, endpoint: string, fn: () => Promise<any>) {
  const apiKey = c.get('apiKey') as ApiKeyContext
  const generationId = await createGeneration(apiKey.id, endpoint)
  const deducted = await deductOneCredit(apiKey.id, generationId)

  if (!deducted) {
    await markFailedNoRefund(generationId, 'insufficient_credits')
    return jsonError(c, 402, 'insufficient_credits', '额度不足。')
  }

  try {
    const result = await fn()
    await markSucceeded(generationId)
    return c.json(result)
  } catch (error: any) {
    console.error('Model request failed:', error?.message || error)
    await markFailedAndRefund(apiKey.id, generationId, error?.message || 'unknown')
    return jsonError(c, 500, 'model_request_failed', `AI 模型调用失败：${error?.message || 'unknown'}`)
  }
}

app.get('/', (c) => {
  return c.json({
    message: 'FloorRoute API is running.',
    routes: ['GET /api/credits', 'POST /api/corner', 'POST /api/search', 'POST /api/path'],
  })
})

app.get('/api/credits', async (c) => {
  const apiKey = c.get('apiKey') as ApiKeyContext
  return c.json({ balance: await getBalance(apiKey.id) })
})

app.post('/api/corner', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', '请求格式错误：缺少 imageDataUrl。')
  }

  return chargeAndRun(c, '/api/corner', async () => {
    const result = await callVisionJson(
      `你是一个文档边角检测器。请找到图片中矩形标牌/平面图的四个角。
返回归一化坐标（0到1），顺序为左上、右上、右下、左下。
只输出 JSON：{"corners":[{"x":0.12,"y":0.05},{"x":0.88,"y":0.06},{"x":0.87,"y":0.92},{"x":0.11,"y":0.91}],"message":"已识别平面图边框。"}
如果无法可靠识别，请仍然返回最接近的四个角。`,
      body.imageDataUrl
    )

    if (!validateCorners(result.corners)) {
      return { corners: fallbackCorners(), message: '未能准确识别平面图边框，请手动调整四角。' }
    }
    return {
      corners: result.corners,
      message: typeof result.message === 'string' ? result.message : '已识别平面图边框。',
    }
  })
})

app.post('/api/search', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', '请求格式错误：缺少 imageDataUrl。')
  }
  if (typeof body.query !== 'string' || body.query.trim() === '') {
    return jsonError(c, 400, 'invalid_request', '请求格式错误：缺少 query。')
  }

  const query = body.query.trim()
  const limit = normalizeLimit(body.limit)

  return chargeAndRun(c, '/api/search', async () => {
    const result = await callVisionJson(
      `你是室内平面图目的地搜索器。
用户给你一张已校正的室内平面图和搜索词：「${query}」。
要求：
1. 只根据图中可见的文字、房间、设施返回匹配候选。
2. 不要编造图中看不到的地点。
3. 最多返回 ${limit} 个。
4. candidates 按 confidence 由高到低排序。
5. 只输出 JSON。
JSON 格式：{"message":"已找到可能匹配的目的地，请选择最准确的一项。","candidates":[{"id":"restroom-east","title":"卫生间","subtitle":"东侧电梯厅旁","confidence":0.92}]}
如果没有匹配，返回：{"message":"未找到匹配的目的地。","candidates":[]}`,
      body.imageDataUrl
    )

    const candidates = validateCandidates(result.candidates, limit)
    return {
      candidates,
      message: typeof result.message === 'string'
        ? result.message
        : candidates.length > 0 ? '已找到可能匹配的目的地，请选择最准确的一项。' : '未找到匹配的目的地。',
    }
  })
})

app.post('/api/path', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', '请求格式错误：缺少 imageDataUrl。')
  }
  if (typeof body.destination !== 'string' || body.destination.trim() === '') {
    return jsonError(c, 400, 'invalid_request', '请求格式错误：缺少 destination。')
  }

  const destination = body.destination.trim()

  return chargeAndRun(c, '/api/path', async () => {
    const resultImageUrl = await callImageEdit(body.imageDataUrl, destination)
    return { resultImageUrl, message: '已生成导航路线。' }
  })
})

// --- Start server ---

await initDb()
console.log(`FloorRoute API starting on port ${PORT}`)
serve({ fetch: app.fetch, port: PORT })
