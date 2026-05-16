# AIuthor — AI Engineer Technical Assessment

**Production-grade multi-agent AI book generation system**

---

## Quick Start (One Command)

```bash
# 1. Clone / navigate to project directory
cd "ai book 2"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
set GROQ_API_KEY=your_key_here      # Windows CMD
# $env:GROQ_API_KEY="your_key_here"  # PowerShell
# export GROQ_API_KEY=your_key_here  # Linux/Mac

# 4. Launch
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

### Docker

```bash
cp .env.example .env          # edit with your GROQ_API_KEY
docker-compose up --build
```

---

## Architecture

```
User Brief (Streamlit)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph StateGraph DAG                │
│                                                     │
│  [1] PlannerAgent      → BookOutline (Pydantic)    │
│       │                                             │
│  [2] ResearcherAgent   → FAISS + BM25 + RRF        │
│       │                                             │
│  [3] WriterAgent       → Chapter prose              │
│       │                                             │
│  [4] HumanizerAgent    → AI-tell removal            │
│       │                                             │
│  [5] EditorAgent       → Pacing + grammar           │
│       │                                             │
│  [6] FactCheckerAgent  → Claim verification         │
│       │                                             │
│  [7] MemoryKeeperAgent → SQLite persistence         │
│       │                                             │
│  [8] AssemblerAgent    → DOCX + PDF                 │
│       │                                             │
│  [9] EvaluationPipeline → LLM-as-judge report      │
└─────────────────────────────────────────────────────┘
        │
        ▼
   outputs/ (DOCX + PDF)
   logs/    (JSONL traces)
   evaluations/ (JSON report)
```

---

## Project Structure

```
ai book 2/
├── agents/             8 agent modules
├── memory/             SQLite persistent store
├── rag/                FAISS + BM25 hybrid retrieval
├── prompts/            All prompt templates
├── schemas/            Pydantic models + LangGraph state
├── orchestration/      LangGraph DAG
├── evaluator/          LLM-as-judge pipeline
├── utils/              Logger + cost tracker
├── frontend/           Streamlit UI
├── logs/               Auto-generated JSONL traces
├── outputs/            Generated DOCX + PDF
├── evaluations/        JSON evaluation reports
├── docs/               Architecture + dossier docs
├── main.py             Entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Agents

| # | Agent | Responsibility |
|---|-------|---------------|
| 1 | **PlannerAgent** | Outline, narrative arc, callbacks, glossary seed, tone blueprint |
| 2 | **ResearcherAgent** | FAISS+BM25 hybrid RAG, fact registry, citations |
| 3 | **WriterAgent** | Chapter prose with callbacks, continuity, memory context |
| 4 | **HumanizerAgent** | AI-tell removal, varied rhythm, emotional cadence |
| 5 | **EditorAgent** | Consistency, pacing, readability, grammar, tone alignment |
| 6 | **FactCheckerAgent** | Claim verification, softening unsupported assertions |
| 7 | **MemoryKeeperAgent** | SQLite persistence, callback repair, glossary updates |
| 8 | **AssemblerAgent** | DOCX + PDF with TOC, front/back matter, page numbers |

---

## Tonalities

| Tone | Style |
|------|-------|
| **Conversational** | Warm, direct — contractions, rhetorical questions, second-person |
| **Academic** | Measured, scholarly — precise terminology, third-person, complex sentences |
| **Storyteller** | Vivid narrative — scenes, metaphors, dramatic rhythm variation |
| **Motivational** | Energetic, inspiring — strong verbs, calls to action, affirmations |
| **Witty** | Clever, dry humor — unexpected analogies, knowing observations |

Tone affects: chapters, foreword, preface, afterword, about author, back cover copy.

---

## Memory System

SQLite tables:
- `fact_registry` — verified/unverified claims with citations
- `character_bible` — entities appearing across chapters
- `concept_bible` — domain concepts with chapter tracking
- `callback_index` — narrative seeds with resolution tracking
- `tone_fingerprint` — running style metrics
- `decision_log` — agent decision audit trail
- `glossary` — defined terms with chapter introduction

**Insertion repair**: when a chapter is regenerated, `repair_callbacks_for_chapter()` automatically marks resolved callbacks and `upsert_glossary()` updates definitions.

---

## RAG System

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost)
- **Vector store**: FAISS IndexFlatIP (cosine similarity via pre-normalized vectors)
- **Sparse retrieval**: Custom BM25 (no external dependency)
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- **Seed corpus**: 20 curated AI/technology knowledge chunks

---

## Observability

Every agent writes structured JSONL to `logs/session_{id}.jsonl`:

```json
{"event": "agent_start", "agent": "WriterAgent", "chapter": 3, "timestamp": "..."}
{"event": "prompt", "agent": "WriterAgent", "system_prompt": "...", "user_prompt": "..."}
{"event": "agent_end", "agent": "WriterAgent", "elapsed_seconds": 4.2, "tokens_used": 820}
```

Cost ledger saved to `logs/cost_ledger_{id}.json`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Structural Completeness | TOC, chapters, front/back matter present |
| Tone Fidelity | Does prose match the chosen tone throughout? |
| AI-Tell Detection | Absence of banned phrases (higher = better) |
| Callback Consistency | Planted callbacks are referenced and resolved |
| Fact Grounding | Claims are supported or appropriately softened |
| Readability | Flow, sentence variety, grade level |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key (console.groq.com) |

---

## Design Decisions

1. **reportlab over WeasyPrint** — Pure Python, no GTK/system deps, works on Windows
2. **Custom BM25** — Avoids `rank_bm25` dependency, same accuracy
3. **LangGraph typed state** — `BookState` TypedDict ensures type safety across all nodes
4. **Pydantic everywhere** — All inter-agent data validated; no regex parsing
5. **Separate Humanizer pass** — Dedicated agent with explicit rule list > hoping Writer avoids AI-tells
6. **SQLite for memory** — Survives restarts; no additional infrastructure needed
7. **Seed corpus** — Ensures RAG always has material even without user-provided documents

## Limitations

- Single LLM (llama-3.1-8b-instant) — production would use specialized models per agent
- No external document ingestion UI (can be added via file upload to `HybridRetriever.index()`)
- PDF TOC page numbers are estimated (reportlab doesn't support automatic TOC page refs)
- Rate limits on Groq free tier may slow generation for 10+ chapter books

## Future Improvements

- Multi-model routing (GPT-4 for Writer, smaller for FactChecker)
- User document upload for RAG corpus enrichment
- Chapter regeneration UI with automatic TOC/glossary repair
- Streaming token-by-token output to frontend
- Export to EPUB format
- Multi-book session management
