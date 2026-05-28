import { randomBytes, createHash, randomUUID } from 'node:crypto'
import { writeFileSync } from 'node:fs'

const raw = randomBytes(24).toString('base64url')
const apiKey = `fr_live_${raw}`

const keyHash = createHash('sha256').update(apiKey).digest('hex')
const apiKeyId = `key_${randomUUID()}`
const now = Math.floor(Date.now() / 1000)
const initialCredits = Number(process.argv[2] || 100)

const sql = `
INSERT INTO api_keys
  (id, key_hash, label, status, created_at)
VALUES
  ('${apiKeyId}', '${keyHash}', 'local test key', 'active', ${now});

INSERT INTO api_key_credit_accounts
  (api_key_id, balance, updated_at)
VALUES
  ('${apiKeyId}', ${initialCredits}, ${now});

INSERT INTO api_key_credit_ledger
  (id, api_key_id, type, amount, ref_id, created_at)
VALUES
  ('ledger_${randomUUID()}', '${apiKeyId}', 'grant', ${initialCredits}, NULL, ${now});
`

writeFileSync('seed-test-key.sql', sql)

console.log('API Key (shown only once):')
console.log(apiKey)
console.log('')
console.log('Generated seed-test-key.sql')
console.log(`Initial credits: ${initialCredits}`)