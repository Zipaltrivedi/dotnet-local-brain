"""SQLite + sqlite-vec database layer.

Stores DotNetRecords in a regular SQLite table and their embeddings
in a sqlite-vec virtual table for similarity search.

Usage:
    db = VectorDB()          # opens/creates the database
    db.insert(record)        # store a record
    db.search_similar(vec)   # find similar records by embedding
    db.get_all()             # retrieve all records
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import sqlite_vec

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import (
    Category,
    CSharpVersion,
    DotNetRecord,
    DotNetVersion,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    csharp_version TEXT NOT NULL,
    dotnet_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    description TEXT NOT NULL,
    code_snippet TEXT NOT NULL,
    legacy_equivalent TEXT NOT NULL,
    nuget_packages TEXT NOT NULL DEFAULT '[]',
    source_url TEXT NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'untested',
    validation_target TEXT NOT NULL DEFAULT '',
    validation_error TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_records_category ON records(category);
CREATE INDEX IF NOT EXISTS idx_records_csharp ON records(csharp_version);
CREATE INDEX IF NOT EXISTS idx_records_dotnet ON records(dotnet_version);
CREATE INDEX IF NOT EXISTS idx_records_validation ON records(validation_status);

CREATE TABLE IF NOT EXISTS crawl_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    url TEXT NOT NULL,
    pages_fetched INTEGER NOT NULL DEFAULT 0,
    features_extracted INTEGER NOT NULL DEFAULT 0,
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# sqlite-vec virtual table (created separately because it uses special syntax)
_VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS record_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


class VectorDB:
    """SQLite + sqlite-vec database for DotNetRecords and embeddings."""

    def __init__(self, config: Optional[SovereignConfig] = None) -> None:
        self.config = config or get_config()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.config.db_path), timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript(_SCHEMA_SQL)
        vec_sql = _VEC_TABLE_SQL.format(dim=self.config.embedding_dim)
        self.conn.execute(vec_sql)
        self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----- Record CRUD -----

    def insert(self, record: DotNetRecord) -> None:
        """Insert or replace a single record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO records
            (id, category, csharp_version, dotnet_version, feature_name,
             description, code_snippet, legacy_equivalent, nuget_packages,
             source_url, validation_status, validation_target, validation_error, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.category.value,
                record.csharp_version.value,
                record.dotnet_version.value,
                record.feature_name,
                record.description,
                record.code_snippet,
                record.legacy_equivalent,
                json.dumps(record.nuget_packages),
                record.source_url,
                record.validation_status.value,
                record.validation_target,
                record.validation_error,
                json.dumps(record.tags),
            ),
        )
        self.conn.commit()

    def insert_batch(self, records: list[DotNetRecord]) -> int:
        """Insert multiple records. Returns count of records inserted."""
        count = 0
        for i, record in enumerate(records):
            try:
                self.conn.execute(
                    """INSERT OR REPLACE INTO records
                    (id, category, csharp_version, dotnet_version, feature_name,
                     description, code_snippet, legacy_equivalent, nuget_packages,
                     source_url, validation_status, validation_target, validation_error, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.id,
                        record.category.value,
                        record.csharp_version.value,
                        record.dotnet_version.value,
                        record.feature_name,
                        record.description,
                        record.code_snippet,
                        record.legacy_equivalent,
                        json.dumps(record.nuget_packages),
                        record.source_url,
                        record.validation_status.value,
                        record.validation_target,
                        record.validation_error,
                        json.dumps(record.tags),
                    ),
                )
                count += 1
            except Exception:
                logger.warning("Failed to insert record: %s", record.feature_name)
            if (i + 1) % 5000 == 0:
                self.conn.commit()
        self.conn.commit()
        return count

    def get_by_id(self, record_id: str) -> Optional[DotNetRecord]:
        """Fetch a single record by ID."""
        row = self.conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_all(
        self,
        category: Optional[Category] = None,
        csharp_version: Optional[CSharpVersion] = None,
        validation_status: Optional[ValidationStatus] = None,
        limit: int = 0,
    ) -> list[DotNetRecord]:
        """Fetch records with optional filters."""
        query = "SELECT * FROM records WHERE 1=1"
        params: list = []

        if category:
            query += " AND category = ?"
            params.append(category.value)
        if csharp_version:
            query += " AND csharp_version = ?"
            params.append(csharp_version.value)
        if validation_status:
            query += " AND validation_status = ?"
            params.append(validation_status.value)

        query += " ORDER BY created_at DESC"
        if limit > 0:
            query += f" LIMIT {limit}"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_validation(
        self,
        record_id: str,
        status: ValidationStatus,
        target: str,
        error: Optional[str] = None,
    ) -> None:
        """Update validation results for a record."""
        self.conn.execute(
            """UPDATE records
            SET validation_status = ?, validation_target = ?, validation_error = ?
            WHERE id = ?""",
            (status.value, target, error, record_id),
        )
        self.conn.commit()

    def count(
        self,
        category: Optional[Category] = None,
        validation_status: Optional[ValidationStatus] = None,
    ) -> int:
        """Count records with optional filters."""
        query = "SELECT COUNT(*) FROM records WHERE 1=1"
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category.value)
        if validation_status:
            query += " AND validation_status = ?"
            params.append(validation_status.value)
        return self.conn.execute(query, params).fetchone()[0]

    # ----- Embedding operations -----

    def store_embedding(self, record_id: str, embedding: list[float]) -> None:
        """Store or update an embedding for a record."""
        self.conn.execute(
            "INSERT OR REPLACE INTO record_embeddings (id, embedding) VALUES (?, ?)",
            (record_id, _serialize_vec(embedding)),
        )
        self.conn.commit()

    def store_embeddings_batch(
        self, pairs: list[tuple[str, list[float]]]
    ) -> int:
        """Store multiple embeddings. Returns count stored."""
        count = 0
        for record_id, embedding in pairs:
            try:
                self.store_embedding(record_id, embedding)
                count += 1
            except Exception:
                logger.warning("Failed to store embedding for %s", record_id)
        return count

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[DotNetRecord, float]]:
        """Find the most similar records by embedding distance.

        Returns list of (record, distance) tuples, sorted by distance ascending.
        """
        rows = self.conn.execute(
            """SELECT r.*, e.distance
            FROM record_embeddings e
            JOIN records r ON r.id = e.id
            WHERE e.embedding MATCH ?
            ORDER BY e.distance
            LIMIT ?""",
            (_serialize_vec(query_embedding), top_k),
        ).fetchall()

        results = []
        for row in rows:
            record = self._row_to_record(row)
            distance = row["distance"]
            results.append((record, distance))
        return results

    # ----- Crawl log -----

    def log_crawl(
        self,
        category: str,
        url: str,
        pages_fetched: int,
        features_extracted: int,
    ) -> None:
        """Record a crawl operation in the audit log."""
        self.conn.execute(
            """INSERT INTO crawl_log (category, url, pages_fetched, features_extracted)
            VALUES (?, ?, ?, ?)""",
            (category, url, pages_fetched, features_extracted),
        )
        self.conn.commit()

    # ----- Stats -----

    def stats(self) -> dict:
        """Return summary statistics about the database."""
        total = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        validated = self.conn.execute(
            "SELECT COUNT(*) FROM records WHERE validation_status = 'compiles'"
        ).fetchone()[0]
        failed = self.conn.execute(
            "SELECT COUNT(*) FROM records WHERE validation_status = 'fails'"
        ).fetchone()[0]
        embedded = self.conn.execute(
            "SELECT COUNT(*) FROM record_embeddings"
        ).fetchone()[0]

        by_category = {}
        for row in self.conn.execute(
            "SELECT category, COUNT(*) as cnt FROM records GROUP BY category"
        ).fetchall():
            by_category[row["category"]] = row["cnt"]

        by_version = {}
        for row in self.conn.execute(
            "SELECT csharp_version, COUNT(*) as cnt FROM records GROUP BY csharp_version"
        ).fetchall():
            by_version[row["csharp_version"]] = row["cnt"]

        return {
            "total_records": total,
            "validated": validated,
            "failed": failed,
            "untested": total - validated - failed,
            "embedded": embedded,
            "by_category": by_category,
            "by_version": by_version,
        }

    # ----- Internal helpers -----

    def _row_to_record(self, row: sqlite3.Row) -> DotNetRecord:
        """Convert a database row to a DotNetRecord."""
        return DotNetRecord(
            id=row["id"],
            category=Category(row["category"]),
            csharp_version=CSharpVersion(row["csharp_version"]),
            dotnet_version=DotNetVersion(row["dotnet_version"]),
            feature_name=row["feature_name"],
            description=row["description"],
            code_snippet=row["code_snippet"],
            legacy_equivalent=row["legacy_equivalent"],
            nuget_packages=json.loads(row["nuget_packages"]),
            source_url=row["source_url"],
            validation_status=ValidationStatus(row["validation_status"]),
            validation_target=row["validation_target"],
            validation_error=row["validation_error"],
            tags=json.loads(row["tags"]),
        )


def _serialize_vec(embedding: list[float]) -> bytes:
    """Serialize a float list to bytes for sqlite-vec."""
    import struct
    return struct.pack(f"{len(embedding)}f", *embedding)
