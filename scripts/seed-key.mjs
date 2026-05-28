import { createClient } from '@libsql/client'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const dbUrl = process.env.DB_URL || 'file:./data/floor-route.db'
const sqlFile = resolve('seed-test-key.sql')

const db = createClient({ url: dbUrl })
const sql = readFileSync(sqlFile, 'utf8')
await db.executeMultiple(sql)
console.log('Key inserted into database.')
