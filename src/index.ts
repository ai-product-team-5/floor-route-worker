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

async function callImageEdit(imageDataUrl: string, destination: string): Promise<string> {
  const prompt = `你正在查看一张室内平面图。你的任务是在图上绘制一条导航路线。

第一步 - 分析平面图：
- 识别所有墙壁（构成房间边界的黑色实线）。
- 识别走廊和可通行区域（房间之间的空白/灰色开放区域）。
- 识别门（墙壁上的缺口/开口）。
- 找到图例（Legend），查看"当前位置"对应的图标/标记样式。
- 在平面图中找到该标记所在的位置，作为路线起点。
- 找到目的地："${destination}"。

第二步 - 规划路线：
- 路线只能经过走廊和门洞。
- 路线绝对不能穿过任何墙壁（黑色实线）。
- 利用走廊网络从起点导航到目的地。
- 选择在可通行区域内的最短路径。

第三步 - 绘制路线：
- 沿规划路线画一条粗红色虚线（---）。
- 线条必须保持在走廊中间，不能接触墙壁。
- 用绿色圆点标记起点。
- 用红色圆点标记目的地。
- 保持原始平面图完全可见，不做任何修改。

重要：墙壁是障碍物。路线必须通过走廊绕过房间，绝不能穿过房间。`

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
    routes: ['GET /api/credits', 'POST /api/corner', 'POST /api/search', 'POST /api/path', 'GET /api/task/:id'],
  })
})

app.get('/api/credits', async (c) => {
  const apiKey = c.get('apiKey') as ApiKeyContext
  return c.json({ balance: await getBalance(apiKey.id) })
})

app.post('/api/corner', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }

  return chargeAndRun(c, '/api/corner', async () => {
    const result = await callVisionJson(
      `You are a floor plan boundary detector. The image shows a photo of an indoor floor plan (possibly mounted on a wall or printed on paper).
Your task: find the four corners of the FLOOR PLAN DIAGRAM AREA ONLY — the region containing room layouts, corridors, and labels. Do NOT include surrounding elements like title bars, headers, footers, legends, or the physical frame/border of the sign.
Return normalized coordinates (0 to 1 relative to the full image), in order: top-left, top-right, bottom-right, bottom-left.
Only output JSON: {"corners":[{"x":0.12,"y":0.15},{"x":0.88,"y":0.15},{"x":0.87,"y":0.85},{"x":0.11,"y":0.84}],"message":"Floor plan area detected."}
If the floor plan fills the entire image with no surrounding elements, return corners near the image edges.`,
      body.imageDataUrl
    )

    if (!validateCorners(result.corners)) {
      return { corners: fallbackCorners(), message: 'Could not accurately detect floor plan area. Please adjust corners manually.' }
    }
    return {
      corners: result.corners,
      message: typeof result.message === 'string' ? result.message : 'Floor plan area detected.',
    }
  })
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
      `You are an indoor floor plan destination search engine.
The user provides a corrected indoor floor plan image and a search query: "${query}".
Requirements:
1. Only return matching candidates based on visible text, rooms, and facilities in the image.
2. Do not fabricate locations not visible in the image.
3. Return at most ${limit} candidates.
4. Sort candidates by confidence from high to low.
5. Only output JSON.
JSON format: {"message":"Found matching destinations.","candidates":[{"id":"restroom-east","title":"Restroom","subtitle":"Near east elevator","confidence":0.92}]}
If no match, return: {"message":"No matching destination found.","candidates":[]}`,
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

// --- Async path generation ---

app.post('/api/path', async (c) => {
  const body = await c.req.json().catch(() => null)
  if (!body || !isDataUrl(body.imageDataUrl)) {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing imageDataUrl.')
  }
  if (typeof body.destination !== 'string' || body.destination.trim() === '') {
    return jsonError(c, 400, 'invalid_request', 'Invalid request: missing destination.')
  }

  const apiKey = c.get('apiKey') as ApiKeyContext
  const destination = body.destination.trim()

  // Charge credit upfront
  const generationId = await createGeneration(apiKey.id, '/api/path')
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
  void processPathTask(taskId, apiKey.id, generationId, body.imageDataUrl, destination)

  // Return immediately
  return c.json({ taskId, message: 'Path generation started.' })
})

async function processPathTask(
  taskId: string,
  apiKeyId: string,
  generationId: string,
  imageDataUrl: string,
  destination: string,
) {
  try {
    const resultImageUrl = await callImageEdit(imageDataUrl, destination)
    await markSucceeded(generationId)
    await db.execute({
      sql: 'UPDATE tasks SET status = \'completed\', result_image_url = ?, finished_at = ? WHERE id = ?',
      args: [resultImageUrl, nowUnix(), taskId],
    })
  } catch (error: any) {
    const errorMsg = error?.message || 'unknown'
    console.error('Path generation failed:', errorMsg)
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
      resultImageUrl: row.result_image_url as string,
      message: 'Navigation route generated.',
    })
  }

  if (status === 'failed') {
    return c.json({
      status: 'failed',
      message: `Generation failed: ${row.error_message || 'unknown'}`,
    })
  }

  return c.json({ status: 'processing', message: 'Generating route...' })
})

// --- Start server ---

await initDb()
console.log(`FloorRoute API starting on port ${PORT}`)
const server = serve({ fetch: app.fetch, port: PORT })
;(server as any).requestTimeout = 600_000
;(server as any).headersTimeout = 600_000
;(server as any).keepAliveTimeout = 600_000
server.on('connection', (socket: import('net').Socket) => {
  socket.setKeepAlive(true, 30_000)
})
