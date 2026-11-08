"""Investigation case management backed by SQLite.

Every change to a case (creation, assignment, status change, note) is written
to an append-only ``audit_log`` table with actor, timestamp and detail, so the
history of each decision can be reconstructed for auditors and regulators.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATUSES = ["New", "In review", "Awaiting information", "Escalated to MLRO",
            "SAR filed", "Closed - no further action"]
OPEN_STATUSES = STATUSES[:4]
TRANSITIONS = {
    "New": ["In review", "Closed - no further action"],
    "In review": ["Awaiting information", "Escalated to MLRO", "Closed - no further action"],
    "Awaiting information": ["In review", "Escalated to MLRO", "Closed - no further action"],
    "Escalated to MLRO": ["SAR filed", "Closed - no further action", "In review"],
    "SAR filed": [],
    "Closed - no further action": ["In review"],   # re-open, always audited
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, month TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    risk_score REAL, risk_band TEXT, queue_rank INTEGER,
    status TEXT NOT NULL, assigned_to TEXT,
    network_cluster TEXT, typologies TEXT, rules_fired TEXT,
    indicator_ids TEXT, drivers_json TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
    author TEXT NOT NULL, created_at TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
    ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, details TEXT
);
CREATE INDEX IF NOT EXISTS ix_cases_month ON cases(month);
CREATE INDEX IF NOT EXISTS ix_audit_case ON audit_log(case_id);
CREATE INDEX IF NOT EXISTS ix_notes_case ON notes(case_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class CaseStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # -- reads ----------------------------------------------------------------
    def cases(self, month: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM cases", ()
        if month:
            q, args = q + " WHERE month = ?", (month,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY month DESC, queue_rank", c, params=args)

    def case(self, case_id: str) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            return dict(r) if r else None

    def notes(self, case_id: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM notes WHERE case_id = ? ORDER BY created_at, note_id",
                                     c, params=(case_id,))

    def audit(self, case_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM audit_log", ()
        if case_id:
            q, args = q + " WHERE case_id = ?", (case_id,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY ts, event_id", c, params=args)

    # -- writes (all audited) -------------------------------------------------
    def _log(self, con, case_id, actor, action, details, ts=None):
        con.execute("INSERT INTO audit_log(case_id, ts, actor, action, details) VALUES (?,?,?,?,?)",
                    (case_id, ts or _now(), actor, action, details))

    def create(self, row: dict, actor: str = "system", ts: str | None = None) -> None:
        ts = ts or _now()
        row = {**row, "created_at": ts, "updated_at": ts, "status": row.get("status", "New")}
        cols = ",".join(row)
        with self._conn() as c:
            c.execute(f"INSERT INTO cases({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
            self._log(c, row["case_id"], actor, "case_created",
                      f"Queued at rank {row.get('queue_rank')} with risk score {row.get('risk_score')}", ts)

    def assign(self, case_id: str, analyst: str, actor: str, ts: str | None = None) -> None:
        ts = ts or _now()
        with self._conn() as c:
            prev = c.execute("SELECT assigned_to FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if prev is None:
                raise KeyError(case_id)
            c.execute("UPDATE cases SET assigned_to=?, updated_at=? WHERE case_id=?", (analyst, ts, case_id))
            self._log(c, case_id, actor, "assigned", f"{prev[0] or 'unassigned'} -> {analyst}", ts)

    def set_status(self, case_id: str, new_status: str, actor: str, reason: str = "",
                   ts: str | None = None) -> None:
        if new_status not in STATUSES:
            raise ValueError(f"Unknown status {new_status!r}")
        ts = ts or _now()
        with self._conn() as c:
            prev = c.execute("SELECT status FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if prev is None:
                raise KeyError(case_id)
            if new_status not in TRANSITIONS[prev[0]]:
                raise ValueError(f"Cannot move a case from '{prev[0]}' to '{new_status}'")
            if new_status in ("Closed - no further action", "SAR filed") and not reason.strip():
                raise ValueError("A rationale is required to close a case or record a SAR")
            c.execute("UPDATE cases SET status=?, updated_at=? WHERE case_id=?", (new_status, ts, case_id))
            self._log(c, case_id, actor, "status_changed",
                      f"{prev[0]} -> {new_status}" + (f" | {reason}" if reason else ""), ts)

    def add_note(self, case_id: str, author: str, text: str, ts: str | None = None) -> None:
        if not text.strip():
            raise ValueError("Note is empty")
        ts = ts or _now()
        with self._conn() as c:
            c.execute("INSERT INTO notes(case_id, author, created_at, text) VALUES (?,?,?,?)",
                      (case_id, author, ts, text.strip()))
            self._log(c, case_id, author, "note_added", text.strip()[:120], ts)


def drivers_to_json(drivers: pd.DataFrame) -> str:
    return json.dumps([
        {"label": r.label, "value": round(float(r.value), 3), "shap": round(float(r.shap), 4),
         "direction": r.direction} for r in drivers.itertuples()])
