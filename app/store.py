"""Submission state store — pluggable persistence for HITL pause/resume.

Why this exists: the One AI data-agent contract is one-shot (query in, response
out) and cannot hold a long human-review pause. So review state lives HERE, keyed
by submission_id, and a separate resume call continues the workflow.

Two implementations behind one interface (injected, never a global):
  - InMemoryStateStore : default; zero dependencies; fine for dev/standalone.
  - PostgresStateStore  : production; NEVAG_DATABASE_URL + psycopg. Stores the
                          workflow state as JSONB keyed by submission_id.

build_state_store(settings) picks Postgres when a database_url is configured and
psycopg is importable, otherwise falls back to in-memory (with a clear warning).
The brain only ever sees the StateStore interface, so swapping is transparent.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Protocol, runtime_checkable

from .config import Settings


@runtime_checkable
class StateStore(Protocol):
    def save(self, submission_id: str, state: dict) -> None: ...
    def load(self, submission_id: str) -> Optional[dict]: ...
    def exists(self, submission_id: str) -> bool: ...


class InMemoryStateStore:
    """Process-local dict store. State does not survive a restart."""

    def __init__(self) -> None:
        self._states: Dict[str, dict] = {}

    def save(self, submission_id: str, state: dict) -> None:
        # Round-trip through JSON so callers can't mutate stored state by reference
        # (and so we fail fast here if a non-serializable value sneaks in).
        self._states[submission_id] = json.loads(json.dumps(state, default=str))

    def load(self, submission_id: str) -> Optional[dict]:
        state = self._states.get(submission_id)
        return json.loads(json.dumps(state, default=str)) if state is not None else None

    def exists(self, submission_id: str) -> bool:
        return submission_id in self._states


class SqliteStateStore:
    """Local file-backed store using stdlib sqlite3 (no dependency). State is
    stored as a JSON column keyed by submission_id. Good default for local runs
    that need persistence across restarts without standing up Postgres."""

    _TABLE = "nevag_submission_state"

    def __init__(self, path: str) -> None:
        import sqlite3

        self._path = path
        self._sqlite3 = sqlite3
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._TABLE} ("
                f"submission_id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                f"updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            conn.commit()

    def _connect(self):
        return self._sqlite3.connect(self._path)

    def save(self, submission_id: str, state: dict) -> None:
        payload = json.dumps(state, default=str)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self._TABLE} (submission_id, state) VALUES (?, ?) "
                f"ON CONFLICT(submission_id) DO UPDATE SET state=excluded.state, "
                f"updated_at=datetime('now')",
                (submission_id, payload),
            )
            conn.commit()

    def load(self, submission_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT state FROM {self._TABLE} WHERE submission_id=?", (submission_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def exists(self, submission_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self._TABLE} WHERE submission_id=?", (submission_id,)
            ).fetchone()
        return row is not None


def _sqlite_path(database_url: str) -> str:
    """Parse a sqlite URL to a path. sqlite:///abs/file.db, sqlite://rel.db,
    sqlite://:memory:."""
    rest = database_url.split("sqlite://", 1)[1]
    if rest == ":memory:":
        return ":memory:"
    return rest[1:] if rest.startswith("/") else rest


class PostgresStateStore:
    """PostgreSQL-backed store (JSONB). Integration point for production.

    Not exercised without a live database + `psycopg` installed; construction
    raises if either is missing so build_state_store can fall back cleanly.
    """

    _TABLE = "nevag_submission_state"

    def __init__(self, settings: Settings) -> None:
        if not settings.database_url:
            raise RuntimeError("PostgresStateStore requires NEVAG_DATABASE_URL")
        try:
            import psycopg  # noqa: F401  (lazy; only needed for this backend)
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(f"psycopg not installed: {exc}") from exc
        self._dsn = settings.database_url
        self._ensure_schema()

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    submission_id TEXT PRIMARY KEY,
                    state         JSONB NOT NULL,
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()

    def save(self, submission_id: str, state: dict) -> None:
        payload = json.dumps(state, default=str)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._TABLE} (submission_id, state, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (submission_id)
                DO UPDATE SET state = EXCLUDED.state, updated_at = now()
                """,
                (submission_id, payload),
            )
            conn.commit()

    def load(self, submission_id: str) -> Optional[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT state FROM {self._TABLE} WHERE submission_id = %s",
                (submission_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def exists(self, submission_id: str) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self._TABLE} WHERE submission_id = %s",
                (submission_id,),
            )
            return cur.fetchone() is not None


def build_state_store(settings: Settings) -> StateStore:
    """Factory chosen by NEVAG_DATABASE_URL. Local-first: sqlite:// for a file-
    backed store; postgres(ql):// for Postgres; otherwise in-memory. Any failure
    degrades to in-memory (with a warning) so local runs never hard-fail."""
    url = getattr(settings, "database_url", None)
    if not url:
        return InMemoryStateStore()

    lowered = url.lower()
    try:
        if lowered.startswith("sqlite://"):
            return SqliteStateStore(_sqlite_path(url))
        if lowered.startswith(("postgres://", "postgresql://", "postgresql+")):
            return PostgresStateStore(settings)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Falling back to InMemoryStateStore (state store unavailable: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
    return InMemoryStateStore()
