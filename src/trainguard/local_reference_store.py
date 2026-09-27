"""SQLite object-store reference for local, same-host protocol fault tests only.

This models durable conditional object operations across local processes. SQLite WAL
requires clients on one host and is not a remote object service, a production
checkpoint backend, or evidence of cross-host fencing and power-loss durability.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from trainguard.remote_protocol import PreconditionFailed, StoredObject


class LocalReferenceObjectStore:
    """Persist the ObjectStore test contract in a same-host SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as database:
            mode = database.execute("PRAGMA journal_mode").fetchone()[0]
            if mode.lower() != "wal":
                mode = database.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise RuntimeError("local reference store requires same-host SQLite WAL")
            with database:
                database.execute("BEGIN IMMEDIATE")
                database.execute(
                    "CREATE TABLE IF NOT EXISTS revisions ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                    "next_revision INTEGER NOT NULL CHECK(next_revision >= 1))"
                )
                database.execute(
                    "INSERT OR IGNORE INTO revisions(singleton, next_revision) VALUES (1, 1)"
                )
                database.execute(
                    "CREATE TABLE IF NOT EXISTS objects ("
                    "key TEXT PRIMARY KEY, data BLOB NOT NULL, "
                    "revision INTEGER NOT NULL UNIQUE)"
                )

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        database.execute("PRAGMA busy_timeout=30000")
        database.execute("PRAGMA synchronous=FULL")
        return database

    @staticmethod
    def _stored(row: tuple[bytes, int] | None) -> StoredObject | None:
        if row is None:
            return None
        data, revision = row
        return StoredObject(bytes(data), f"revision-{revision}")

    def get(self, key: str) -> StoredObject | None:
        with closing(self._connect()) as database:
            row = database.execute(
                "SELECT data, revision FROM objects WHERE key=?", (key,)
            ).fetchone()
        return self._stored(row)

    def list(self, prefix: str) -> tuple[str, ...]:
        with closing(self._connect()) as database:
            keys = (row[0] for row in database.execute("SELECT key FROM objects"))
            return tuple(sorted(key for key in keys if key.startswith(prefix)))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> StoredObject:
        if if_none_match == (if_match is not None):
            raise ValueError("put requires exactly one conditional precondition")
        payload = bytes(data)
        with closing(self._connect()) as database, database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                "SELECT revision FROM objects WHERE key=?", (key,)
            ).fetchone()
            current_etag = None if row is None else f"revision-{row[0]}"
            if if_none_match and row is not None:
                raise PreconditionFailed(f"object already exists: {key}")
            if if_match is not None and current_etag != if_match:
                raise PreconditionFailed(f"object revision changed: {key}")
            revision = database.execute(
                "SELECT next_revision FROM revisions WHERE singleton=1"
            ).fetchone()[0]
            database.execute(
                "UPDATE revisions SET next_revision=? WHERE singleton=1", (revision + 1,)
            )
            database.execute(
                "INSERT INTO objects(key, data, revision) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET data=excluded.data, revision=excluded.revision",
                (key, payload, revision),
            )
        return StoredObject(payload, f"revision-{revision}")

    def delete(self, key: str, *, if_match: str | None = None) -> bool:
        with closing(self._connect()) as database, database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                "SELECT revision FROM objects WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return False
            if if_match is not None and f"revision-{row[0]}" != if_match:
                raise PreconditionFailed(f"object revision changed: {key}")
            database.execute("DELETE FROM objects WHERE key=?", (key,))
        return True
