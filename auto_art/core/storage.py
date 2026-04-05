"""
Evaluation history storage with SQLite backend and content-addressable caching.

Replaces flat JSON attack_memory.json with structured database for:
- Querying past results across evaluations
- Content-addressable caching (hash(model+attack+config) -> cached result)
- Model fingerprinting for cache invalidation
- Cross-project evaluation history

Design patterns:
- Repository pattern for data access
- Content-addressable storage (like Git) for attack result caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(str(Path.home()), ".auto_art", "evaluation_history.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    model_hash TEXT,
    framework TEXT,
    model_type TEXT,
    timestamp TEXT NOT NULL,
    duration_seconds REAL,
    config_json TEXT,
    summary_json TEXT,
    compliance_json TEXT,
    status TEXT DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS attack_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id TEXT NOT NULL,
    attack_name TEXT NOT NULL,
    attack_category TEXT,
    success_rate REAL,
    perturbation_norm REAL,
    num_samples INTEGER,
    duration_seconds REAL,
    config_json TEXT,
    result_json TEXT,
    cache_key TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(evaluation_id)
);

CREATE TABLE IF NOT EXISTS attack_cache (
    cache_key TEXT PRIMARY KEY,
    attack_name TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 1,
    size_bytes INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_fingerprints (
    model_hash TEXT PRIMARY KEY,
    model_path TEXT NOT NULL,
    framework TEXT,
    file_size INTEGER,
    created_at TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_attack_results_eval ON attack_results(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_attack_results_name ON attack_results(attack_name);
CREATE INDEX IF NOT EXISTS idx_attack_cache_model ON attack_cache(model_hash);
CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model_hash);
"""


@dataclass
class EvaluationRecord:
    evaluation_id: str
    model_path: str
    model_hash: Optional[str]
    framework: Optional[str]
    model_type: Optional[str]
    timestamp: str
    duration_seconds: Optional[float]
    config: Dict[str, Any]
    summary: Dict[str, Any]
    compliance: Dict[str, Any]
    status: str = "completed"


@dataclass
class AttackResultRecord:
    evaluation_id: str
    attack_name: str
    attack_category: Optional[str]
    success_rate: float
    perturbation_norm: Optional[float]
    num_samples: Optional[int]
    duration_seconds: Optional[float]
    config: Dict[str, Any]
    result: Dict[str, Any]
    cache_key: Optional[str] = None


class EvaluationStore:
    """SQLite-backed evaluation history store.

    Provides structured storage for evaluation results, attack outcomes,
    and content-addressable caching for expensive attack computations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -------------------------------------------------------------------
    # Evaluation CRUD
    # -------------------------------------------------------------------

    def save_evaluation(self, record: EvaluationRecord) -> None:
        """Save an evaluation record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO evaluations
                   (evaluation_id, model_path, model_hash, framework, model_type,
                    timestamp, duration_seconds, config_json, summary_json,
                    compliance_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.evaluation_id, record.model_path, record.model_hash,
                    record.framework, record.model_type, record.timestamp,
                    record.duration_seconds, json.dumps(record.config, default=str),
                    json.dumps(record.summary, default=str),
                    json.dumps(record.compliance, default=str),
                    record.status,
                ),
            )

    def get_evaluation(self, evaluation_id: str) -> Optional[EvaluationRecord]:
        """Get a single evaluation by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            if row:
                return self._row_to_evaluation(row)
        return None

    def list_evaluations(
        self,
        model_path: Optional[str] = None,
        framework: Optional[str] = None,
        limit: int = 100,
    ) -> List[EvaluationRecord]:
        """List evaluations with optional filtering."""
        query = "SELECT * FROM evaluations WHERE 1=1"
        params: List[Any] = []

        if model_path:
            query += " AND model_path = ?"
            params.append(model_path)
        if framework:
            query += " AND framework = ?"
            params.append(framework)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_evaluation(r) for r in rows]

    @staticmethod
    def _row_to_evaluation(row: sqlite3.Row) -> EvaluationRecord:
        return EvaluationRecord(
            evaluation_id=row["evaluation_id"],
            model_path=row["model_path"],
            model_hash=row["model_hash"],
            framework=row["framework"],
            model_type=row["model_type"],
            timestamp=row["timestamp"],
            duration_seconds=row["duration_seconds"],
            config=json.loads(row["config_json"] or "{}"),
            summary=json.loads(row["summary_json"] or "{}"),
            compliance=json.loads(row["compliance_json"] or "{}"),
            status=row["status"],
        )

    # -------------------------------------------------------------------
    # Attack results
    # -------------------------------------------------------------------

    def save_attack_result(self, record: AttackResultRecord) -> None:
        """Save an attack result."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO attack_results
                   (evaluation_id, attack_name, attack_category, success_rate,
                    perturbation_norm, num_samples, duration_seconds,
                    config_json, result_json, cache_key, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.evaluation_id, record.attack_name, record.attack_category,
                    record.success_rate, record.perturbation_norm, record.num_samples,
                    record.duration_seconds, json.dumps(record.config, default=str),
                    json.dumps(record.result, default=str), record.cache_key,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_attack_history(
        self,
        attack_name: Optional[str] = None,
        model_hash: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query attack result history."""
        query = """SELECT ar.*, e.model_hash, e.model_path, e.framework
                   FROM attack_results ar
                   JOIN evaluations e ON ar.evaluation_id = e.evaluation_id
                   WHERE 1=1"""
        params: List[Any] = []

        if attack_name:
            query += " AND ar.attack_name = ?"
            params.append(attack_name)
        if model_hash:
            query += " AND e.model_hash = ?"
            params.append(model_hash)

        query += " ORDER BY ar.timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------
    # Content-addressable cache
    # -------------------------------------------------------------------

    @staticmethod
    def compute_cache_key(
        model_hash: str,
        attack_name: str,
        config: Dict[str, Any],
    ) -> str:
        """Compute content-addressable cache key: hash(model+attack+config)."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        content = f"{model_hash}:{attack_name}:{config_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached attack result."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM attack_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row:
                conn.execute(
                    """UPDATE attack_cache
                       SET accessed_at = ?, access_count = access_count + 1
                       WHERE cache_key = ?""",
                    (datetime.now(timezone.utc).isoformat(), cache_key),
                )
                return json.loads(row["result_json"])
        return None

    def cache_result(
        self,
        cache_key: str,
        attack_name: str,
        model_hash: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Store an attack result in the cache."""
        result_json = json.dumps(result, default=str)
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO attack_cache
                   (cache_key, attack_name, model_hash, config_hash,
                    result_json, created_at, accessed_at, access_count, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (cache_key, attack_name, model_hash, config_hash,
                 result_json, now, now, len(result_json)),
            )

    def evict_cache(self, max_entries: int = 10000, max_age_days: int = 90) -> int:
        """Evict old/unused cache entries. Returns number of entries removed."""
        with self._connect() as conn:
            cutoff = datetime.now(timezone.utc).isoformat()
            # Remove entries older than max_age_days that haven't been accessed recently
            result = conn.execute(
                """DELETE FROM attack_cache
                   WHERE julianday(?) - julianday(accessed_at) > ?""",
                (cutoff, max_age_days),
            )
            age_evicted = result.rowcount

            # If still over limit, remove least recently accessed
            count = conn.execute("SELECT COUNT(*) FROM attack_cache").fetchone()[0]
            lru_evicted = 0
            if count > max_entries:
                excess = count - max_entries
                conn.execute(
                    """DELETE FROM attack_cache WHERE cache_key IN
                       (SELECT cache_key FROM attack_cache
                        ORDER BY accessed_at ASC LIMIT ?)""",
                    (excess,),
                )
                lru_evicted = excess

            total = age_evicted + lru_evicted
            if total > 0:
                logger.info(f"Cache eviction: {total} entries removed "
                            f"({age_evicted} aged, {lru_evicted} LRU)")
            return total

    # -------------------------------------------------------------------
    # Model fingerprinting
    # -------------------------------------------------------------------

    @staticmethod
    def compute_model_hash(model_path: str) -> str:
        """Compute hash of model file for fingerprinting."""
        sha = hashlib.sha256()
        path = Path(model_path)
        if not path.exists():
            return hashlib.sha256(model_path.encode()).hexdigest()[:16]

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]

    def save_model_fingerprint(
        self,
        model_hash: str,
        model_path: str,
        framework: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a model fingerprint."""
        file_size = Path(model_path).stat().st_size if Path(model_path).exists() else 0
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_fingerprints
                   (model_hash, model_path, framework, file_size, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (model_hash, model_path, framework, file_size,
                 datetime.now(timezone.utc).isoformat(),
                 json.dumps(metadata or {}, default=str)),
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM attack_cache").fetchone()[0]
            total_size = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM attack_cache"
            ).fetchone()[0]
            total_hits = conn.execute(
                "SELECT COALESCE(SUM(access_count), 0) FROM attack_cache"
            ).fetchone()[0]
            total_evals = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]

        return {
            "cache_entries": total,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "total_hits": total_hits,
            "total_evaluations": total_evals,
        }
