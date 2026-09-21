import json
import sqlite3
from datetime import UTC, datetime


class Store:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            # Previous process cannot resume running tasks safely.
            for job_id, payload in db.execute("SELECT id,payload FROM jobs").fetchall():
                job = json.loads(payload)
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", error="服务重启，请重新提交分析。")
                    db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(job), job_id))

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save(self, job):
        job["updated_at"] = datetime.now(UTC).isoformat()
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM jobs ORDER BY rowid DESC LIMIT 50").fetchall()
        return [json.loads(row[0]) for row in rows]
