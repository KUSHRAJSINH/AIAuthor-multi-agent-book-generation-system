# AIuthor — Memory Schema Documentation

## Overview

Persistent memory is stored in SQLite (`memory/aiuthor_memory.db`).
Every table includes a `session_id` column to support multi-session isolation.

---

## Tables

### `fact_registry`
Stores verified and unverified factual claims from the ResearcherAgent and FactCheckerAgent.

| Column | Type | Description |
|--------|------|-------------|
| fact_id | TEXT PK | Unique ID e.g. `f_ch1_001` |
| session_id | TEXT | Session isolation key |
| chapter | INTEGER | Chapter where fact appears |
| claim | TEXT | The factual assertion |
| supported | INTEGER | 1=verified, 0=unverified |
| citations | TEXT | JSON array of Citation objects |
| softened | TEXT | Hedged version if unsupported |
| created_at | TEXT | ISO timestamp |

---

### `character_bible`
Tracks named entities (people, organizations, case studies) across chapters.

| Column | Type | Description |
|--------|------|-------------|
| name | TEXT PK | Entity name |
| session_id | TEXT | Session isolation key |
| role | TEXT | Role in the narrative |
| first_appearance | INTEGER | Chapter number |
| description | TEXT | Full description |
| traits | TEXT | JSON array of trait strings |
| created_at | TEXT | ISO timestamp |

---

### `concept_bible`
Tracks domain concepts introduced across chapters to prevent re-explaining them.

| Column | Type | Description |
|--------|------|-------------|
| concept | TEXT PK | Concept name |
| session_id | TEXT | Session isolation key |
| explanation | TEXT | One-paragraph explanation |
| chapter_introduced | INTEGER | First chapter where concept appears |
| complexity | TEXT | simple / intermediate / advanced |
| created_at | TEXT | ISO timestamp |

---

### `callback_index`
Tracks narrative callbacks — concepts/events planted in early chapters and resolved later.

| Column | Type | Description |
|--------|------|-------------|
| callback_id | TEXT | e.g. `cb_001` |
| session_id | TEXT | Session isolation key |
| introduced_in | INTEGER | Chapter where callback is planted |
| description | TEXT | What the callback refers to |
| referenced_in | TEXT | JSON array of chapter numbers |
| resolved | INTEGER | 1=resolved, 0=open |
| created_at | TEXT | ISO timestamp |

**Repair logic**: `repair_callbacks_for_chapter(n)` sets `resolved=1` for any callback whose `referenced_in` includes `n`.

---

### `tone_fingerprint`
Running average of style metrics computed by the MemoryKeeperAgent.

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT PK | Session isolation key |
| tone_name | TEXT | Selected tone |
| avg_sentence_length | REAL | Average words per sentence |
| contraction_ratio | REAL | Fraction of sentences with contractions |
| question_frequency | REAL | Fraction of sentences that are questions |
| exclamation_frequency | REAL | Fraction with exclamation marks |
| second_person_ratio | REAL | Fraction using "you" |
| ai_tell_count | INTEGER | Cumulative AI-tell violations |
| sample_sentences | TEXT | JSON array of 5 example sentences |
| updated_at | TEXT | ISO timestamp |

---

### `decision_log`
Audit trail of agent decisions with rationale.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT | Session isolation key |
| agent | TEXT | Agent name |
| chapter | INTEGER | Chapter context |
| decision | TEXT | What was decided |
| rationale | TEXT | Why it was decided |
| created_at | TEXT | ISO timestamp |

---

### `glossary`
Domain-specific terms with definitions. Updated by MemoryKeeperAgent after each chapter.

| Column | Type | Description |
|--------|------|-------------|
| term | TEXT | Glossary term |
| session_id | TEXT | Session isolation key |
| definition | TEXT | Definition text |
| chapter_intro | INTEGER | Chapter where first introduced |
| tone_variant | TEXT | Optional tone-specific phrasing |
| created_at | TEXT | ISO timestamp |

**Composite PK**: `(term, session_id)` — prevents duplicate terms per session.
**Upsert logic**: `ON CONFLICT DO UPDATE SET definition=excluded.definition`

---

## Insertion Repair Flow

When a chapter is regenerated:

```
1. FactCheckerAgent re-runs → fact_registry updated via upsert
2. MemoryKeeperAgent re-runs → glossary terms upserted (no duplicates)
3. repair_callbacks_for_chapter(n) called → resolved flags updated
4. Assembler re-runs → TOC rebuilt from chapter list (dynamic)
5. Glossary section rebuilt from store.get_glossary()
```

No manual intervention required.

---

## Pydantic ↔ SQLite Mapping

| Pydantic Model | SQLite Table |
|----------------|-------------|
| FactEntry | fact_registry |
| CharacterEntry | character_bible |
| ConceptEntry | concept_bible |
| CallbackReference | callback_index |
| ToneFingerprint | tone_fingerprint |
| DecisionLogEntry | decision_log |
| GlossaryEntry | glossary |
