"""Disk-based cache with SQLite index and TTL."""
from __future__ import annotations
import json, logging, sqlite3, time
from pathlib import Path

logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, cache_dir="~/.videoscan/cache", max_size_gb=5, enabled=True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_size_gb = max_size_gb
        self._db = None
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for d in ("downloads", "frames", "results"):
                (self.cache_dir / d).mkdir(exist_ok=True)
            self._db = sqlite3.connect(str(self.cache_dir / "cache.db"))
            self._db.execute("CREATE TABLE IF NOT EXISTS cache_entries (key TEXT PRIMARY KEY, type TEXT NOT NULL, path TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL)")
            self._db.commit()

    def store_result(self, video_id, params_hash, data, ttl=604800):
        if not self.enabled or not self._db: return
        key = f"{video_id}_{params_hash}"
        path = self.cache_dir / "results" / f"{key}.json"
        path.write_text(json.dumps(data))
        now = time.time()
        self._db.execute("INSERT OR REPLACE INTO cache_entries VALUES (?,?,?,?,?)", (key, "result", str(path), now, now + ttl))
        self._db.commit()

    def get_result(self, video_id, params_hash, force_refresh=False):
        if not self.enabled or not self._db or force_refresh: return None
        key = f"{video_id}_{params_hash}"
        row = self._db.execute("SELECT path, expires_at FROM cache_entries WHERE key=? AND type=?", (key, "result")).fetchone()
        if not row: return None
        path, expires_at = row
        if time.time() > expires_at:
            self._db.execute("DELETE FROM cache_entries WHERE key=?", (key,)); self._db.commit()
            return None
        try: return json.loads(Path(path).read_text())
        except: self._db.execute("DELETE FROM cache_entries WHERE key=?", (key,)); self._db.commit(); return None

    def store_frames(self, video_id, frames, ttl=86400):
        if not self.enabled or not self._db: return
        d = self.cache_dir / "frames" / video_id
        d.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(frames):
            (d / f"frame_{i:04d}.jpg").write_bytes(data)
        now = time.time()
        self._db.execute("INSERT OR REPLACE INTO cache_entries VALUES (?,?,?,?,?)", (f"frames_{video_id}", "frames", str(d), now, now + ttl))
        self._db.commit()

    def get_frames(self, video_id, force_refresh=False):
        if not self.enabled or not self._db or force_refresh: return None
        row = self._db.execute("SELECT path, expires_at FROM cache_entries WHERE key=? AND type=?", (f"frames_{video_id}", "frames")).fetchone()
        if not row: return None
        path, expires_at = row
        if time.time() > expires_at:
            self._db.execute("DELETE FROM cache_entries WHERE key=?", (f"frames_{video_id}",)); self._db.commit()
            return None
        d = Path(path)
        if not d.exists(): return None
        return [f.read_bytes() for f in sorted(d.glob("frame_*.jpg"), key=lambda f: f.name)]

    def cleanup(self):
        if not self.enabled or not self._db: return
        for key, path in self._db.execute("SELECT key, path FROM cache_entries WHERE expires_at < ?", (time.time(),)).fetchall():
            p = Path(path)
            if p.is_file(): p.unlink(missing_ok=True)
            elif p.is_dir():
                import shutil; shutil.rmtree(p, ignore_errors=True)
            self._db.execute("DELETE FROM cache_entries WHERE key=?", (key,))
        self._db.commit()
