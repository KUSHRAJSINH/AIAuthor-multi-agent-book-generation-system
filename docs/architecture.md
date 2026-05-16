# AIuthor — Architecture Document

## System Overview

AIuthor is a production-grade multi-agent book generation system. A user provides a brief (topic, tone, chapter count), and the system orchestrates 8 specialized AI agents through a LangGraph DAG to produce a publication-ready book in DOCX and PDF formats.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Brief Form  │  │ Config Panel │  │ Evaluation Dashboard   │ │
│  │ (textarea)  │  │ (tone/count) │  │ (metrics + downloads)  │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │ run_pipeline()
┌────────────────────────────▼─────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATION                        │
│                                                                  │
│  BookState (TypedDict) flows through each node:                  │
│                                                                  │
│  planner → researcher → writer → humanizer → editor              │
│      → fact_checker → memory_keeper → assembler → evaluator      │
│                                                                  │
│  Each node: receives BookState, returns updated BookState        │
│  Retry logic: exponential backoff (2^attempt) per agent          │
└──────────┬──────────────┬──────────────┬──────────────┬──────────┘
           │              │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌────▼──────┐
    │  GROQ LLM   │ │  RAG Layer │ │  SQLite   │ │  Output   │
    │ (8 agents)  │ │ FAISS+BM25 │ │  Memory   │ │ DOCX+PDF  │
    └─────────────┘ └────────────┘ └───────────┘ └───────────┘
```

---

## Data Flow

### 1. Planning Phase
```
UserBrief → PlannerAgent → BookOutline{
    title, subtitle, chapters[ChapterPlan], 
    glossary_seed[GlossaryEntry],
    callback_index[CallbackReference],
    tone: ToneMetadata
}
```

### 2. Research Phase
```
For each chapter in BookOutline.chapters:
    query = chapter.title + chapter.key_concepts
    dense_results = FAISS.search(embed(query), top_k=10)
    sparse_results = BM25.retrieve(query, top_k=10)
    fused = RRF(dense_results, sparse_results)[:5]
    
    ResearcherAgent(fused) → ResearchPacket{
        key_facts[], citations[], fact_registry[]
    }
```

### 3. Writing Phase
```
For each chapter:
    context = {
        research_facts: ResearchPacket.key_facts,
        open_callbacks: CallbackIndex.get_open_callbacks(),
        established_concepts: ConceptBible[:5]
    }
    WriterAgent(context) → raw_prose
```

### 4. Refinement Phase
```
raw_prose → HumanizerAgent → humanized_prose
humanized_prose → EditorAgent → edited_prose  
edited_prose → FactCheckerAgent → final_prose
```

### 5. Memory Phase
```
For each final_prose:
    MemoryKeeperAgent extracts:
        - new concepts → concept_bible table
        - new glossary terms → glossary table
        - callback updates → callback_index table
        - tone metrics → tone_fingerprint table
    repair_callbacks_for_chapter(n)
```

### 6. Assembly Phase
```
AssemblerAgent:
    LLM generates: foreword, preface, afterword, about_author, back_cover
    python-docx: builds DOCX with TOC, styles, sections
    reportlab: builds PDF with page numbers, styled TOC table
```

### 7. Evaluation Phase
```
EvaluationPipeline:
    chapter_excerpts = first 2000 chars × first 3 chapters
    LLM judge scores 6 metrics (0-10)
    EvaluationReport → evaluations/eval_{timestamp}.json
```

---

## State Management

`BookState` (TypedDict) is the single source of truth flowing through the LangGraph DAG:

```python
class BookState(TypedDict, total=False):
    user_brief: str
    tone_name: str
    chapter_count: int
    chapter_length: int
    author_name: str
    outline: Dict          # BookOutline.model_dump()
    research_packets: Dict # {chapter_num: ResearchPacket.model_dump()}
    chapters: Dict         # {chapter_num: ChapterContent.model_dump()}
    callback_index: Dict
    fact_registry: Dict
    glossary: List[Dict]
    tone_fingerprint: Dict
    docx_path: str
    pdf_path: str
    agent_logs: List[str]
    errors: List[str]
    evaluation_report: Dict
```

Each agent receives the full state and returns it with its additions merged in.

---

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | `llama-3.1-8b-instant` via Groq | Fast (280+ tok/s), free tier, production-quality |
| Orchestration | LangGraph | Native TypedDict state, DAG visualization, streaming |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local inference, no API cost, 384-dim, fast |
| Vector DB | FAISS IndexFlatIP | Pure Python install, no server, cosine via normalized vecs |
| Sparse Retrieval | Custom BM25 | No external dep, same accuracy as rank_bm25 |
| Memory | SQLite | No infrastructure, survives restarts, ACID guarantees |
| DOCX | python-docx | Mature, stable, complex formatting support |
| PDF | reportlab | Pure Python, no system deps (avoids WeasyPrint/GTK issues on Windows) |
| Validation | Pydantic v2 | Type-safe inter-agent contracts, no regex parsing |
| Frontend | Streamlit | Rapid deployment, native widgets for forms/downloads |

---

## Error Handling

Every agent implements:
1. **Retry loop** (max 3 attempts) with exponential backoff
2. **JSON extraction** — strips markdown fences, finds first `{` to last `}`
3. **Fallback returns** — graceful degradation, never crashes the pipeline
4. **Logging** — all errors written to JSONL log before fallback

The LangGraph `recursion_limit=50` prevents infinite loops.

---

## Security

- API key loaded from environment variable, never hardcoded
- Streamlit input sanitized before passing to LLM
- SQLite uses parameterized queries throughout (no SQL injection risk)
- Output files written to local `outputs/` directory only
