import { createClient } from "@libsql/client";
const db = createClient({ url: "file:./data/floor-route.db" });
const hash = "812634787db6afa602f47a59c67639d0edd6ef1431080e9a94453dcb3326a64e";
const now = Math.floor(Date.now() / 1000);
await db.batch([
  { sql: "INSERT OR IGNORE INTO api_keys (id, key_hash, label, status, created_at) VALUES (?, ?, ?, ?, ?)", args: ["key_local_test", hash, "local test", "active", now] },
  { sql: "INSERT OR IGNORE INTO api_key_credit_accounts (api_key_id, balance, updated_at) VALUES (?, ?, ?)", args: ["key_local_test", 100, now] },
]);
console.log("Seeded OK");
