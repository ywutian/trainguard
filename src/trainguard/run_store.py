"""SQLite index for runs, attempts, checkpoints, and recovery decisions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trainguard.events import utc_now


class RunStore:
    def __init__(self, path: Path) -> None:
        self.database = sqlite3.connect(path)
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA foreign_keys=ON")
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
        self.database.commit()

    def close(self) -> None:
        self.database.close()

    def create_run(self, run_id: str, fingerprint: str, started_at: str) -> None:
        self.database.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            (run_id, "RUNNING", fingerprint, started_at, started_at),
        )
        self.database.commit()

    def set_run_status(self, run_id: str, status: str) -> None:
        self.database.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
            (status, utc_now(), run_id),
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
