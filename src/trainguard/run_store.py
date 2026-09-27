"""SQLite index for runs, attempts, checkpoints, and recovery decisions."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from pathlib import Path

from trainguard.events import utc_now


class RunStore:
    def __init__(self, path: Path, *, existing_only: bool = False) -> None:
        path = Path(path).absolute()
        if not path.exists() and not path.is_symlink():
            if existing_only:
                raise FileNotFoundError("run index is missing")
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            os.close(descriptor)
        self.database = self._open_verified(path)
        try:
            if existing_only:
                self._verify_existing_schema()
                return
            self._initialize_schema()
        except Exception:
            self.database.close()
            raise

    @staticmethod
    def _open_verified(path: Path) -> sqlite3.Connection:
        """Open an existing regular index without following a replaced path."""
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("run index is not a regular file")
            expected = path.parent.resolve() / path.name
            database = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
            try:
                listed = database.execute("PRAGMA database_list").fetchone()
                after = path.lstat()
                if (
                    not stat.S_ISREG(after.st_mode)
                    or after.st_nlink != 1
                    or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                    or listed is None
                    or Path(listed[2]).resolve() != expected
                ):
                    raise ValueError("run index changed while opening")
            except Exception:
                database.close()
                raise
            return database
        finally:
            os.close(descriptor)

    def _verify_existing_schema(self) -> None:
        result = self.database.execute("PRAGMA quick_check").fetchone()
        if result != ("ok",):
            raise sqlite3.DatabaseError("run index integrity check failed")
        for query in (
            (
                "SELECT run_id, status, config_fingerprint, started_at, updated_at, "
                "evidence_schema_version, measurement_sha256, environment_sha256 "
                "FROM runs LIMIT 0"
            ),
            (
                "SELECT attempt_id, run_id, number, status, resume_checkpoint, resume_step, "
                "resume_consumed_batches, pid, pid_identity, started_at, finished_at, "
                "exit_code, reason FROM attempts LIMIT 0"
            ),
            (
                "SELECT path, run_id, attempt_id, global_step, status, reason, checked_at "
                "FROM checkpoints LIMIT 0"
            ),
            (
                "SELECT id, run_id, from_attempt, to_attempt, checkpoint_path, resume_step, "
                "recomputed_steps, created_at FROM recoveries LIMIT 0"
            ),
        ):
            self.database.execute(query)
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA foreign_keys=ON")

    def _initialize_schema(self) -> None:
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA foreign_keys=ON")
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                evidence_schema_version INTEGER NOT NULL DEFAULT 1,
                environment_sha256 TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                number INTEGER NOT NULL,
                status TEXT NOT NULL,
                resume_checkpoint TEXT,
                resume_step INTEGER NOT NULL,
                pid INTEGER,
                pid_identity TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                exit_code INTEGER,
                reason TEXT,
                UNIQUE(run_id, number)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                path TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                attempt_id TEXT,
                global_step INTEGER,
                status TEXT NOT NULL,
                reason TEXT,
                checked_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                from_attempt TEXT NOT NULL,
                to_attempt TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                resume_step INTEGER NOT NULL,
                recomputed_steps INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in self.database.execute("PRAGMA table_info(attempts)")}
        if "resume_consumed_batches" not in columns:
            self.database.execute(
                "ALTER TABLE attempts ADD COLUMN resume_consumed_batches INTEGER NOT NULL DEFAULT 0"
            )
            self.database.execute("UPDATE attempts SET resume_consumed_batches=resume_step")
        run_columns = {row[1] for row in self.database.execute("PRAGMA table_info(runs)")}
        if "evidence_schema_version" not in run_columns:
            self.database.execute(
                "ALTER TABLE runs ADD COLUMN evidence_schema_version INTEGER NOT NULL DEFAULT 1"
            )
        if "measurement_sha256" not in run_columns:
            self.database.execute("ALTER TABLE runs ADD COLUMN measurement_sha256 TEXT")
        if "environment_sha256" not in run_columns:
            self.database.execute("ALTER TABLE runs ADD COLUMN environment_sha256 TEXT")
        self.database.commit()

    def close(self) -> None:
        self.database.close()

    def create_run(
        self, run_id: str, fingerprint: str, started_at: str, *, evidence_schema_version: int = 1,
        environment_sha256: str | None = None,
    ) -> None:
        if evidence_schema_version == 3 and (
            not isinstance(environment_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", environment_sha256) is None
        ):
            raise ValueError("new run evidence requires a runtime identity digest")
        self.database.execute(
            "INSERT INTO runs (run_id, status, config_fingerprint, started_at, updated_at, "
            "evidence_schema_version, environment_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, "RUNNING", fingerprint, started_at, started_at,
                evidence_schema_version, environment_sha256,
            ),
        )
        self.database.commit()

    def run_identity(self, run_id: str) -> sqlite3.Row | None:
        return self.database.execute(
            "SELECT run_id, config_fingerprint, started_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()

    def set_run_status(self, run_id: str, status: str) -> None:
        self.database.execute(
            "UPDATE runs SET status=?, updated_at=?, measurement_sha256=NULL WHERE run_id=?",
            (status, utc_now(), run_id),
        )
        self.database.commit()

    def set_run_success(self, run_id: str, measurement_sha256: str | None) -> None:
        self.database.execute(
            "UPDATE runs SET status='SUCCEEDED', updated_at=?, measurement_sha256=? "
            "WHERE run_id=?",
            (utc_now(), measurement_sha256, run_id),
        )
        self.database.commit()

    def attempts(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self.database.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY number", (run_id,)
            )
        )

    def start_attempt(
        self,
        run_id: str,
        attempt_id: str,
        number: int,
        resume_checkpoint: str | None,
        resume_step: int,
        resume_consumed_batches: int | None = None,
    ) -> None:
        self.database.execute(
            """INSERT INTO attempts
            (attempt_id, run_id, number, status, resume_checkpoint, resume_step, started_at, resume_consumed_batches)
            VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?)""",
            (attempt_id, run_id, number, resume_checkpoint, resume_step, utc_now(),
             resume_step if resume_consumed_batches is None else resume_consumed_batches),
        )
        self.database.commit()

    def set_pid(self, attempt_id: str, pid: int, identity: str) -> None:
        self.database.execute(
            "UPDATE attempts SET pid=?, pid_identity=? WHERE attempt_id=?",
            (pid, identity, attempt_id),
        )
        self.database.commit()

    def finish_attempt(
        self, attempt_id: str, status: str, exit_code: int | None, reason: str
    ) -> None:
        self.database.execute(
            """UPDATE attempts SET status=?, finished_at=?, exit_code=?, reason=?
            WHERE attempt_id=?""",
            (status, utc_now(), exit_code, reason, attempt_id),
        )
        self.database.commit()

    def discard_unlaunched_attempt(self, attempt_id: str) -> None:
        self.database.execute("DELETE FROM recoveries WHERE to_attempt=?", (attempt_id,))
        self.database.execute(
            "DELETE FROM attempts WHERE attempt_id=? AND pid IS NULL",
            (attempt_id,),
        )
        self.database.commit()

    def record_checkpoint(
        self,
        path: str,
        run_id: str,
        attempt_id: str | None,
        global_step: int | None,
        status: str,
        reason: str | None,
    ) -> None:
        self.record_checkpoints([(path, run_id, attempt_id, global_step, status, reason)])

    def record_checkpoints(self, records: list[tuple]) -> None:
        with self.database:
            self.database.executemany(
                """INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                attempt_id=excluded.attempt_id, global_step=excluded.global_step,
                status=excluded.status, reason=excluded.reason, checked_at=excluded.checked_at""",
                [(*record, utc_now()) for record in records],
            )

    def record_recovery(
        self,
        run_id: str,
        from_attempt: str,
        to_attempt: str,
        checkpoint_path: str,
        resume_step: int,
        recomputed_steps: int,
    ) -> None:
        self.database.execute(
            """INSERT INTO recoveries
            (run_id, from_attempt, to_attempt, checkpoint_path, resume_step,
             recomputed_steps, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                from_attempt,
                to_attempt,
                checkpoint_path,
                resume_step,
                recomputed_steps,
                utc_now(),
            ),
        )
        self.database.commit()
