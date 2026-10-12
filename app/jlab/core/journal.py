"""Append-only journal: SQLite (WAL) + hash chain + triggers that forbid
UPDATE and DELETE at the database level, not merely by convention.

Every consequential act in the app writes here: a lab run, a card admitted
through the gate, an egress attempt, a theta event, an LLM call.  The chain
makes silent rewriting of history detectable: entry i stores
``sha256(prev_hash || canonical_json(payload))``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from . import config

GENESIS = "0" * 64

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS journal (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    kind      TEXT    NOT NULL,
    actor     TEXT    NOT NULL,
    payload   TEXT    NOT NULL,
    prev_hash TEXT    NOT NULL,
    hash      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS journal_kind_idx ON journal(kind);
CREATE INDEX IF NOT EXISTS journal_ts_idx   ON journal(ts);

CREATE TRIGGER IF NOT EXISTS journal_no_update
BEFORE UPDATE ON journal
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;

CREATE TRIGGER IF NOT EXISTS journal_no_delete
BEFORE DELETE ON journal
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
"""


def canonical(payload: dict) -> str:
    """Deterministic JSON so the hash chain is reproducible across runs."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class Journal:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.state_dir() / "journal.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writing ------------------------------------------------------------
    def append(self, kind: str, payload: dict, actor: str = "app") -> dict:
        with self._lock:
            cur = self._conn.execute("SELECT hash FROM journal ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev = row[0] if row else GENESIS
            ts = time.time()
            body = canonical({"ts": ts, "kind": kind, "actor": actor, "payload": payload})
            h = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            self._conn.execute(
                "INSERT INTO journal(ts,kind,actor,payload,prev_hash,hash) VALUES (?,?,?,?,?,?)",
                (ts, kind, actor, canonical(payload), prev, h),
            )
            self._conn.commit()
            return {"ts": ts, "kind": kind, "actor": actor, "hash": h, "prev_hash": prev}

    # -- reading ------------------------------------------------------------
    def tail(self, n: int = 50, kind: str | None = None) -> list[dict]:
        q = "SELECT id,ts,kind,actor,payload,prev_hash,hash FROM journal"
        args: tuple = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY id DESC LIMIT ?"
        args = args + (n,)
        rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "ts": r[1], "kind": r[2], "actor": r[3],
                "payload": json.loads(r[4]), "prev_hash": r[5], "hash": r[6],
            })
        return out

    def count(self, kind: str | None = None) -> int:
        if kind:
            return self._conn.execute(
                "SELECT COUNT(*) FROM journal WHERE kind=?", (kind,)).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0]

    def verify(self) -> dict:
        """Recompute the whole chain.  Returns the first break, if any."""
        prev = GENESIS
        n = 0
        for r in self._conn.execute(
                "SELECT id,ts,kind,actor,payload,prev_hash,hash FROM journal ORDER BY id"):
            _id, ts, kind, actor, payload, prev_hash, h = r
            body = canonical({"ts": ts, "kind": kind, "actor": actor,
                              "payload": json.loads(payload)})
            expect = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            if prev_hash != prev or h != expect:
                return {"ok": False, "broken_at": _id, "checked": n}
            prev = h
            n += 1
        return {"ok": True, "checked": n, "head": prev}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_default: Journal | None = None


def get() -> Journal:
    global _default
    if _default is None:
        _default = Journal()
    return _default


def log(kind: str, payload: dict, actor: str = "app") -> dict:
    return get().append(kind, payload, actor=actor)
