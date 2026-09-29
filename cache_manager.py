#!/usr/bin/env python3
"""
High-Performance Persistent Cache for Math Explorer.
Thread-safe SQLite caching for:
- JEV Intent Routing
- BM25 / PageIndex Retrieval Candidates
- JEV Node Relevance Scoring
- Solver Outputs (SymPy, Violetto, Astra)
- Verification Verdicts
"""

import os
import json
import sqlite3
import hashlib
import threading
from typing import Optional, Any, Dict
from config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "math_explorer_cache.db")

class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CacheManager, cls).__new__(cls)
                cls._instance._init_db(db_path)
            return cls._instance

    def _init_db(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.local = threading.local()
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, cache_key)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON kv_cache(namespace)")
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self.local, "conn") or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path, timeout=10.0)
        return self.local.conn

    @staticmethod
    def hash_key(*args) -> str:
        h = hashlib.sha256()
        for a in args:
            if isinstance(a, dict) or isinstance(a, list):
                h.update(json.dumps(a, sort_keys=True).encode("utf-8"))
            else:
                h.update(str(a).strip().lower().encode("utf-8"))
        return h.hexdigest()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM kv_cache WHERE namespace = ? AND cache_key = ?", (namespace, key))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            return None

    def set(self, namespace: str, key: str, value: Any):
        try:
            payload = json.dumps(value)
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO kv_cache (namespace, cache_key, payload) VALUES (?, ?, ?)",
                (namespace, key, payload)
            )
            conn.commit()
        except Exception as e:
            pass

    def clear_namespace(self, namespace: str):
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM kv_cache WHERE namespace = ?", (namespace,))
            conn.commit()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, int]:
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT namespace, count(*) FROM kv_cache GROUP BY namespace")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception:
            return {}

# Singleton instance
cache = CacheManager()
