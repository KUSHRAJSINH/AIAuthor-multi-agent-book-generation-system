"""
SQLite-backed persistent memory store for the AIuthor system.

Tables:
- fact_registry
- character_bible
- concept_bible
- callback_index
- tone_fingerprint
- decision_log
- glossary
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path("memory")
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "aiuthor_memory.db"

DDL = """
CREATE TABLE IF NOT EXISTS fact_registry (
    fact_id     TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    chapter     INTEGER DEFAULT 0,
    claim       TEXT NOT NULL,
    supported   INTEGER DEFAULT 0,
    citations   TEXT DEFAULT '[]',
    softened    TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS character_bible (
    name            TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    role            TEXT,
    first_appearance INTEGER DEFAULT 0,
    description     TEXT,
    traits          TEXT DEFAULT '[]',
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS concept_bible (
    concept         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    explanation     TEXT,
    chapter_introduced INTEGER DEFAULT 0,
    complexity      TEXT DEFAULT 'intermediate',
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS callback_index (
    callback_id         TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    introduced_in       INTEGER DEFAULT 0,
    description         TEXT,
    referenced_in       TEXT DEFAULT '[]',
    resolved            INTEGER DEFAULT 0,
    created_at          TEXT,
    PRIMARY KEY (callback_id, session_id)
);

CREATE TABLE IF NOT EXISTS tone_fingerprint (
    session_id              TEXT PRIMARY KEY,
    tone_name               TEXT,
    avg_sentence_length     REAL DEFAULT 0,
    contraction_ratio       REAL DEFAULT 0,
    question_frequency      REAL DEFAULT 0,
    exclamation_frequency   REAL DEFAULT 0,
    second_person_ratio     REAL DEFAULT 0,
    ai_tell_count           INTEGER DEFAULT 0,
    sample_sentences        TEXT DEFAULT '[]',
    updated_at              TEXT
);

CREATE TABLE IF NOT EXISTS decision_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    agent       TEXT,
    chapter     INTEGER DEFAULT 0,
    decision    TEXT,
    rationale   TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS glossary (
    term            TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    definition      TEXT,
    chapter_intro   INTEGER DEFAULT 0,
    tone_variant    TEXT,
    created_at      TEXT,
    PRIMARY KEY (term, session_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Thread-safe SQLite memory store. Use one instance per session."""

    def __init__(self, session_id: str, db_path: Optional[Path] = None):
        self.session_id = session_id
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    # ------------------------------------------------------------------ Facts

    def upsert_fact(self, fact_id: str, claim: str, supported: bool,
                    citations: List[Dict], chapter: int = 0,
                    softened: Optional[str] = None) -> None:
        sql = """
        INSERT INTO fact_registry
            (fact_id, session_id, chapter, claim, supported, citations, softened, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fact_id) DO UPDATE SET
            supported=excluded.supported,
            citations=excluded.citations,
            softened=excluded.softened
        """
        with self._conn() as conn:
            conn.execute(sql, (
                fact_id, self.session_id, chapter, claim,
                int(supported), json.dumps(citations), softened, _now()
            ))

    def get_facts(self, chapter: Optional[int] = None) -> List[Dict]:
        if chapter is not None:
            sql = "SELECT * FROM fact_registry WHERE session_id=? AND chapter=?"
            params = (self.session_id, chapter)
        else:
            sql = "SELECT * FROM fact_registry WHERE session_id=?"
            params = (self.session_id,)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- Callbacks

    def upsert_callback(self, callback_id: str, introduced_in: int,
                        description: str, referenced_in: List[int],
                        resolved: bool = False) -> None:
        sql = """
        INSERT INTO callback_index
            (callback_id, session_id, introduced_in, description, referenced_in, resolved, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(callback_id, session_id) DO UPDATE SET
            referenced_in=excluded.referenced_in,
            resolved=excluded.resolved
        """
        with self._conn() as conn:
            conn.execute(sql, (
                callback_id, self.session_id, introduced_in, description,
                json.dumps(referenced_in), int(resolved), _now()
            ))

    def get_callbacks(self, chapter: Optional[int] = None) -> List[Dict]:
        if chapter is not None:
            sql = "SELECT * FROM callback_index WHERE session_id=? AND introduced_in<=?"
            params = (self.session_id, chapter)
        else:
            sql = "SELECT * FROM callback_index WHERE session_id=?"
            params = (self.session_id,)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["referenced_in"] = json.loads(d.get("referenced_in", "[]"))
            result.append(d)
        return result

    def repair_callbacks_for_chapter(self, chapter_num: int) -> None:
        """Mark callbacks that should be resolved by this chapter as resolved."""
        callbacks = self.get_callbacks()
        for cb in callbacks:
            refs = cb.get("referenced_in", [])
            if chapter_num in refs and not cb["resolved"]:
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE callback_index SET resolved=1 WHERE callback_id=? AND session_id=?",
                        (cb["callback_id"], self.session_id)
                    )

    # -------------------------------------------------------------- Glossary

    def upsert_glossary(self, term: str, definition: str,
                        chapter_intro: int = 0, tone_variant: str = "") -> None:
        sql = """
        INSERT INTO glossary (term, session_id, definition, chapter_intro, tone_variant, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(term, session_id) DO UPDATE SET
            definition=excluded.definition,
            tone_variant=excluded.tone_variant
        """
        with self._conn() as conn:
            conn.execute(sql, (term, self.session_id, definition, chapter_intro, tone_variant, _now()))

    def get_glossary(self) -> List[Dict]:
        sql = "SELECT * FROM glossary WHERE session_id=? ORDER BY term ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, (self.session_id,)).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- Concepts

    def upsert_concept(self, concept: str, explanation: str,
                       chapter_introduced: int, complexity: str = "intermediate") -> None:
        sql = """
        INSERT INTO concept_bible (concept, session_id, explanation, chapter_introduced, complexity, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(concept) DO UPDATE SET explanation=excluded.explanation
        """
        with self._conn() as conn:
            conn.execute(sql, (concept, self.session_id, explanation, chapter_introduced, complexity, _now()))

    def get_concepts(self) -> List[Dict]:
        sql = "SELECT * FROM concept_bible WHERE session_id=? ORDER BY chapter_introduced"
        with self._conn() as conn:
            rows = conn.execute(sql, (self.session_id,)).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------- Tone Fingerprint

    def save_tone_fingerprint(self, tone_name: str, data: Dict[str, Any]) -> None:
        sql = """
        INSERT INTO tone_fingerprint
            (session_id, tone_name, avg_sentence_length, contraction_ratio,
             question_frequency, exclamation_frequency, second_person_ratio,
             ai_tell_count, sample_sentences, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            tone_name=excluded.tone_name,
            avg_sentence_length=excluded.avg_sentence_length,
            contraction_ratio=excluded.contraction_ratio,
            question_frequency=excluded.question_frequency,
            exclamation_frequency=excluded.exclamation_frequency,
            second_person_ratio=excluded.second_person_ratio,
            ai_tell_count=excluded.ai_tell_count,
            sample_sentences=excluded.sample_sentences,
            updated_at=excluded.updated_at
        """
        with self._conn() as conn:
            conn.execute(sql, (
                self.session_id, tone_name,
                data.get("avg_sentence_length", 0.0),
                data.get("contraction_ratio", 0.0),
                data.get("question_frequency", 0.0),
                data.get("exclamation_frequency", 0.0),
                data.get("second_person_ratio", 0.0),
                data.get("ai_tell_count", 0),
                json.dumps(data.get("sample_sentences", [])),
                _now()
            ))

    def get_tone_fingerprint(self) -> Optional[Dict]:
        sql = "SELECT * FROM tone_fingerprint WHERE session_id=?"
        with self._conn() as conn:
            row = conn.execute(sql, (self.session_id,)).fetchone()
        if row:
            d = dict(row)
            d["sample_sentences"] = json.loads(d.get("sample_sentences", "[]"))
            return d
        return None

    # ------------------------------------------------------------ Decision Log

    def log_decision(self, agent: str, decision: str, rationale: str, chapter: int = 0) -> None:
        sql = """
        INSERT INTO decision_log (session_id, agent, chapter, decision, rationale, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (self.session_id, agent, chapter, decision, rationale, _now()))

    def get_decisions(self) -> List[Dict]:
        sql = "SELECT * FROM decision_log WHERE session_id=? ORDER BY id"
        with self._conn() as conn:
            rows = conn.execute(sql, (self.session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ Characters

    def upsert_character(self, name: str, role: str, first_appearance: int,
                         description: str, traits: List[str]) -> None:
        sql = """
        INSERT INTO character_bible
            (name, session_id, role, first_appearance, description, traits, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET description=excluded.description, traits=excluded.traits
        """
        with self._conn() as conn:
            conn.execute(sql, (name, self.session_id, role, first_appearance,
                               description, json.dumps(traits), _now()))

    def get_characters(self) -> List[Dict]:
        sql = "SELECT * FROM character_bible WHERE session_id=?"
        with self._conn() as conn:
            rows = conn.execute(sql, (self.session_id,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["traits"] = json.loads(d.get("traits", "[]"))
            result.append(d)
        return result
