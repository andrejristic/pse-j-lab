"""SQLite storage for cards, with FTS5 search when the build supports it.

FTS5 is standard in Debian/Ubuntu python3-sqlite builds but the app must not
die if it is missing, so the index is optional and search falls back to LIKE.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..core import config
from .cards import Card, CardKind

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    claim       TEXT NOT NULL,
    topic       TEXT,
    body        TEXT,
    run_id      TEXT,
    sha256      TEXT,
    reference   TEXT,
    tags        TEXT,
    created     REAL,
    updated     REAL,
    author      TEXT,
    lineage     TEXT,
    attacks     TEXT,
    status_note TEXT,
    fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS cards_kind_idx  ON cards(kind);
CREATE INDEX IF NOT EXISTS cards_topic_idx ON cards(topic);
CREATE UNIQUE INDEX IF NOT EXISTS cards_fp_idx ON cards(fingerprint);

CREATE TRIGGER IF NOT EXISTS cards_no_delete
BEFORE DELETE ON cards
BEGIN SELECT RAISE(ABORT, 'cards are never deleted; refute them instead'); END;
"""

_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    id UNINDEXED, claim, body, topic, tags, tokenize='unicode61'
);
"""


class CardStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config.vault_dir() / "cards.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self.fts = True
        try:
            self._conn.executescript(_FTS)
        except sqlite3.OperationalError:
            self.fts = False
        self._conn.commit()

    # -- writing (only the gate should call this) ---------------------------
    def _insert(self, c: Card) -> Card:
        with self._lock:
            self._conn.execute(
                "INSERT INTO cards (id,kind,claim,topic,body,run_id,sha256,reference,"
                "tags,created,updated,author,lineage,attacks,status_note,fingerprint) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (c.id, c.kind, c.claim, c.topic, c.body, c.run_id, c.sha256,
                 c.reference, json.dumps(c.tags), c.created, c.updated, c.author,
                 json.dumps(c.lineage), json.dumps(c.attacks), c.status_note,
                 c.fingerprint()))
            if self.fts:
                self._conn.execute(
                    "INSERT INTO cards_fts (id,claim,body,topic,tags) VALUES (?,?,?,?,?)",
                    (c.id, c.claim, c.body, c.topic, " ".join(c.tags)))
            self._conn.commit()
        return c

    def _update_status(self, card_id: str, kind: str, note: str,
                       attack: dict | None = None) -> Card | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
            if row is None:
                return None
            attacks = json.loads(row["attacks"] or "[]")
            if attack:
                attacks.append(attack)
            import time as _t
            self._conn.execute(
                "UPDATE cards SET kind=?, status_note=?, attacks=?, updated=? WHERE id=?",
                (kind, note, json.dumps(attacks), _t.time(), card_id))
            self._conn.commit()
        return self.get(card_id)

    # -- reading -------------------------------------------------------------
    def _row_to_card(self, row) -> Card:
        return Card(id=row["id"], kind=row["kind"], claim=row["claim"],
                    topic=row["topic"] or "", body=row["body"] or "",
                    run_id=row["run_id"] or "", sha256=row["sha256"] or "",
                    reference=row["reference"] or "",
                    tags=json.loads(row["tags"] or "[]"),
                    created=row["created"] or 0.0, updated=row["updated"] or 0.0,
                    author=row["author"] or "", lineage=json.loads(row["lineage"] or "[]"),
                    attacks=json.loads(row["attacks"] or "[]"),
                    status_note=row["status_note"] or "")

    def get(self, card_id: str) -> Card | None:
        row = self._conn.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
        return self._row_to_card(row) if row else None

    def by_fingerprint(self, fp: str) -> Card | None:
        row = self._conn.execute("SELECT * FROM cards WHERE fingerprint=?", (fp,)).fetchone()
        return self._row_to_card(row) if row else None

    def list(self, kind: str | None = None, topic: str | None = None,
             limit: int = 500) -> list:
        q = "SELECT * FROM cards"
        cond, args = [], []
        if kind:
            cond.append("kind=?"); args.append(kind)
        if topic:
            cond.append("topic=?"); args.append(topic)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY updated DESC LIMIT ?"
        args.append(limit)
        return [self._row_to_card(r) for r in self._conn.execute(q, args)]

    def search(self, text: str, limit: int = 50) -> list:
        text = (text or "").strip()
        if not text:
            return []
        if self.fts:
            try:
                rows = self._conn.execute(
                    "SELECT c.* FROM cards_fts f JOIN cards c ON c.id=f.id "
                    "WHERE cards_fts MATCH ? LIMIT ?", (text, limit)).fetchall()
                return [self._row_to_card(r) for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{text}%"
        rows = self._conn.execute(
            "SELECT * FROM cards WHERE claim LIKE ? OR body LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()
        return [self._row_to_card(r) for r in rows]

    def counts(self) -> dict:
        out = {k: 0 for k in CardKind.ALL}
        for r in self._conn.execute("SELECT kind, COUNT(*) c FROM cards GROUP BY kind"):
            out[r["kind"]] = r["c"]
        out["total"] = sum(out[k] for k in CardKind.ALL)
        return out

    def close(self):
        with self._lock:
            self._conn.close()


_store: CardStore | None = None


def get_store() -> CardStore:
    global _store
    if _store is None:
        _store = CardStore()
    return _store
