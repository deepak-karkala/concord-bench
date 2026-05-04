import asyncio
import json
import sqlite3
from pathlib import Path

from concord.schemas.episode import EpisodeLog

CACHE_DIR = Path("outputs/.cache")
CACHE_DB = CACHE_DIR / "llm_cache.db"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    _ensure_cache_dir()
    db = sqlite3.connect(str(CACHE_DB))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            temperature REAL NOT NULL,
            seed INTEGER NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    return db


class CacheLLMCalls:
    def __init__(self):
        self._db = _get_db()

    def get(self, model: str, prompt_hash: str, temperature: float, seed: int) -> dict | None:
        key = f"{model}:{prompt_hash}:{temperature}:{seed}"
        row = self._db.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, model: str, prompt_hash: str, temperature: float, seed: int, response: dict) -> None:
        key = f"{model}:{prompt_hash}:{temperature}:{seed}"
        self._db.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, model, prompt_hash, temperature, seed, response_json) VALUES (?, ?, ?, ?, ?, ?)",
            (key, model, prompt_hash, temperature, seed, json.dumps(response)),
        )
        self._db.commit()

    def close(self):
        self._db.close()
