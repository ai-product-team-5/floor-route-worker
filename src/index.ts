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

async function callImageEdit(imageDataUrl: string, destination: string): Promise<string> {
  const prompt = `你正在查看一张室内平面图。你的任务是在图上绘制一条从当前位置到目的地的导航路线。

## 分析平面图

1. 识别结构元素：
   - 墙壁：构成房间边界的黑色实线（不可穿越）
   - 走廊：房间之间的空白/灰色开放通行区域
   - 门：墙壁上的缺口或开口（可通行）
2. 定位起点：
   - 查看图例（Legend）中"当前位置"对应的图标样式（可能是星形、箭头、圆点或人形标记）
   - 在平面图主体中找到该标记的位置，即为起点
3. 定位目的地："${destination}"

## 规划路线

- 路线只能经过走廊和门洞，绝不能穿过墙壁
- 转弯时沿走廊方向做直角转弯，不要斜穿区域
- 目的地是房间时，终点设为该房间的门（不进入房间内部）
- 如果目标房间有多扇门，选择从起点出发路径最短的那扇门作为终点
- 选择最短的可通行路径

## 绘制路线

- 画一条粗红色虚线（约 4-6px 宽），沿走廊中线绘制
- 线条与墙壁保持明显间距，不能贴墙
- 起点标记：绿色实心圆点
- 终点标记：红色实心圆点
- 保持原始平面图完全清晰可见，不遮挡任何文字标注

## 关键约束

墙壁是绝对障碍物。即使两点直线距离很近，如果中间有墙，也必须绕行走廊。绝不能穿过房间内部。`

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
