import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { createClient } from '@libsql/client'
import { createHash, randomUUID } from 'node:crypto'
import { imageSize } from 'image-size'

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
 * 从支持的预设尺寸中选择最接近输入宽高比的尺寸。
 * 优先匹配宽高比，其次选面积最接近的。
 */
function pickClosestSize(inputW: number, inputH: number): string {
  const inputRatio = inputW / inputH
  const inputArea = inputW * inputH

  // OpenRouter 支持的预设尺寸（竖向为主，适合平面图）
  const sizes: [number, number][] = [
    [1080, 1080],
    [1080, 1440],
    [1080, 1620],
    [1080, 1920],
    [720, 960],
    [720, 1080],
    [720, 1280],
    [480, 640],
    [480, 720],
  ]

  let best = '1080x1080'
  let bestScore = Infinity
  for (const [w, h] of sizes) {
    const ratio = w / h
    const area = w * h
    // 宽高比差异权重高，面积差异权重低
    const ratioDiff = Math.abs(ratio - inputRatio) * 10
    const areaDiff = Math.abs(area - inputArea) / inputArea
    const score = ratioDiff + areaDiff
    if (score < bestScore) {
      bestScore = score
      best = `${w}x${h}`
    }
  }
  return best
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

async function callImageWallMask(imageDataUrl: string): Promise<string> {
  const prompt = `你正在分析一张室内平面图。你的任务是输出一张**纯黑白二值化的墙体掩码图**，用于后续算法寻路。

## 最重要的约束（违反则输出无效）

输出图必须与输入图**像素级空间对齐**：
- 输出图的分辨率必须与输入图**完全相同**（相同的宽度和高度像素数）
- 每一面墙在输出图中的位置必须与输入图中**完全对应**——如果在输入图中某面墙的左端点在 (100, 200)，那么输出图中该墙的黑色像素也必须从 (100, 200) 开始
- 不要缩放、偏移、旋转、裁剪或添加任何边距
- 把输入图想象成一个图层，你的输出是另一个图层——两者叠加时墙线必须完全重合

## 输出规则

- 仅使用两种颜色：纯黑 (#000000) 和 纯白 (#FFFFFF)，禁止任何灰色/抗锯齿/中间色
- **墙壁 = 黑色**：所有不可通行的实体墙线，保持原图中墙线的精确位置和粗细
- **可通行区域 = 白色**：走廊、房间内部、门洞、室外区域
- 门洞必须保留为白色（让走廊与房间在掩码上保持连通）
- 删除所有文字、数字、房间标签、图例、家具、装饰图标、指北针、比例尺
- 删除所有"当前位置"标记和图例图标
- 不要画任何路径、箭头、起点终点标记
- 不要保留原图的颜色、底纹、阴影

## 质量检查

最终图必须满足：
1. 与原图叠加时，黑色墙线精确覆盖原图的墙体线条
2. 沿任何走廊放一个像素，都能通过白色像素连通到任意房间门口
3. 房间之间通过门洞而不是墙壁连接

只输出图像，不要附带任何文字说明。`

  // 从 data URL 解析图片尺寸，选择最接近的支持尺寸
  const dimensions = parseImageDimensions(imageDataUrl)
  if (!dimensions) {
    throw new Error('无法解析输入图片尺寸')
  }
  const size = pickClosestSize(dimensions.width, dimensions.height)
  console.log(`Input dimensions: ${dimensions.width}x${dimensions.height}, selected size: ${size}`)

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
          size,
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
    const wallMaskDataUrl = await callImageWallMask(imageDataUrl)
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
      message: 'Wall mask generated.',
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

await initDb()
console.log(`FloorRoute API starting on port ${PORT}`)
const server = serve({ fetch: app.fetch, port: PORT })
;(server as any).requestTimeout = 600_000
;(server as any).headersTimeout = 600_000
;(server as any).keepAliveTimeout = 600_000
server.on('connection', (socket: import('net').Socket) => {
  socket.setKeepAlive(true, 30_000)
})
