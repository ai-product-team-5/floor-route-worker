# FloorRoute API Server

方寸识途后端 API 服务。基于 Hono + libsql (SQLite)，部署在 VPS 上。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/credits` | 查询剩余算力 |
| POST | `/api/search` | 在校正后的平面图上搜索目的地候选 |
| POST | `/api/walls` | 异步生成墙体二值掩码（用于前端寻路） |
| POST | `/api/endpoints` | 在平面图上定位起点和终点的归一化坐标 |
| GET | `/api/task/:id` | 查询异步任务状态（墙体掩码） |

所有 `/api/*` 请求需要 `Authorization: Bearer fr_live_xxx` header。

## 部署步骤

### 1. 安装 Node.js

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
```

### 2. 克隆项目

```bash
cd /opt
git clone https://github.com/ai-product-team-5/floor-route-worker.git
cd floor-route-worker
npm install
```

### 3. 配置环境变量

```bash
cp .env.example .env
nano .env
```

填入你的 OpenRouter API Key 和模型配置。

### 4. 初始化数据库并启动

```bash
mkdir -p data
node --env-file=.env --import=tsx src/index.ts
```

### 5. 生成 API Key

```bash
node scripts/create-test-key.mjs
node --env-file=.env --import=tsx -e "
import { createClient } from '@libsql/client';
import { readFileSync } from 'fs';
const db = createClient({ url: process.env.DB_URL });
await db.executeMultiple(readFileSync('seed-test-key.sql', 'utf8'));
console.log('Key inserted.');
"
```

记下打印的 `fr_live_xxx`，这是前端使用的 API Key。

### 6. 后台运行（生产环境）

```bash
npm install -g pm2
pm2 start "node --env-file=.env --import=tsx src/index.ts" --name floor-route-api
pm2 save
pm2 startup
```

## 本地开发

```bash
npm install
mkdir -p data
npm run dev
```

## 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `PORT` | 服务端口 | `8080` |
| `DB_URL` | SQLite 数据库路径 | `file:./data/floor-route.db` |
| `VISION_MODEL_BASE_URL` | 视觉模型 API 地址 | `https://openrouter.ai/api/v1` |
| `VISION_MODEL_API_KEY` | 视觉模型 API Key | `sk-or-xxx` |
| `VISION_MODEL_NAME` | 视觉模型名称 | `qwen/qwen3-vl-235b-a22b-instruct` |
| `IMAGE_MODEL_BASE_URL` | 图像生成模型 API 地址 | `https://openrouter.ai/api/v1` |
| `IMAGE_MODEL_API_KEY` | 图像生成模型 API Key | `sk-or-xxx` |
| `IMAGE_MODEL_NAME` | 图像生成模型名称 | `openai/gpt-5.4-image-2` |

## 技术栈

- [Hono](https://hono.dev/) - Web 框架
- [@libsql/client](https://github.com/tursodatabase/libsql-client-ts) - SQLite 数据库
- [OpenRouter](https://openrouter.ai/) - AI 模型网关
