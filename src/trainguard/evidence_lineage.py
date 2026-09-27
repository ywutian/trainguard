"""Permit evidence-only commits after a verified execution commit."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def require_evidence_only_descendant(root: Path, execution_commit: str,
                                     candidate_commit: str) -> None:
    """Reject any post-execution commit that touches an executable input."""
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (
        execution_commit, candidate_commit
    )):
        raise ValueError("execution or candidate commit identity is malformed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", execution_commit, candidate_commit],
        cwd=root, capture_output=True, check=False,
    )
    if ancestor.returncode:
        raise ValueError("verified execution commit is not a candidate ancestor")
    commits = subprocess.run(
        ["git", "rev-list", f"{execution_commit}..{candidate_commit}"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for commit in commits:
        changed = subprocess.run(
            ["git", "diff-tree", "-m", "--no-commit-id", "--name-only", "-r", "-z",
             "--no-renames", commit],
            cwd=root, capture_output=True, check=True,
        ).stdout
        for raw_path in changed.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8", errors="surrogateescape")
            if path != "README.md" and not path.startswith("docs/"):
                raise ValueError(f"executable or unreviewed file changed after evidence: {path}")
