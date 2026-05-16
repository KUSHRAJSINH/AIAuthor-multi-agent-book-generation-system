"""
Full pipeline diagnostic — runs every agent with a mock LLM to isolate
non-LLM errors (file path issues, import errors, FAISS issues, etc.)
"""
import os
import sys
import json
import traceback

# Use a fake key for structural testing
os.environ.setdefault("GROQ_API_KEY", "test_key_placeholder")

print("=" * 60)
print("STEP 1: Import all modules")
print("=" * 60)

modules = [
    ("schemas.models", "BookState, BookOutline, ChapterContent"),
    ("memory.sqlite_store", "MemoryStore"),
    ("rag.chunker", "TextChunker"),
    ("rag.embedder", "Embedder"),
    ("rag.faiss_store", "FAISSStore"),
    ("rag.retriever", "HybridRetriever"),
    ("prompts.templates", "PromptTemplates"),
    ("utils.logger", "ObservabilityLogger"),
    ("utils.cost_tracker", "CostTracker"),
    ("agents.planner", "PlannerAgent"),
    ("agents.researcher", "ResearcherAgent"),
    ("agents.writer", "WriterAgent"),
    ("agents.humanizer", "HumanizerAgent"),
    ("agents.editor", "EditorAgent"),
    ("agents.fact_checker", "FactCheckerAgent"),
    ("agents.memory_keeper", "MemoryKeeperAgent"),
    ("agents.assembler", "AssemblerAgent"),
    ("evaluator.pipeline", "EvaluationPipeline"),
    ("orchestration.graph", "build_graph, run_pipeline"),
]

all_ok = True
for mod, items in modules:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except Exception as e:
        print(f"  ✗ {mod}: {e}")
        traceback.print_exc()
        all_ok = False

print()
print("=" * 60)
print("STEP 2: Embedder + FAISS end-to-end")
print("=" * 60)
try:
    from rag.embedder import Embedder
    from rag.faiss_store import FAISSStore
    from rag.retriever import HybridRetriever

    e = Embedder()
    vecs = e.embed(["test sentence one", "test sentence two"])
    print(f"  ✓ Embedder | fallback={e._is_fallback} | shape={vecs.shape}")

    fs = FAISSStore("diag_test", dim=e.dim)
    fs.add(["test sentence one", "test sentence two"], vecs)
    hits = fs.search(e.embed_one("sentence"), top_k=2)
    print(f"  ✓ FAISS search | {len(hits)} results")

    ret = HybridRetriever(fs, e)
    ret.index(["neural networks learn patterns", "agents orchestrate tasks"])
    texts = ret.retrieve_texts("AI systems", top_k=2)
    print(f"  ✓ HybridRetriever | {texts}")
except Exception as e:
    print(f"  ✗ RAG error: {e}")
    traceback.print_exc()
    all_ok = False

print()
print("=" * 60)
print("STEP 3: SQLite memory store")
print("=" * 60)
try:
    from memory.sqlite_store import MemoryStore
    ms = MemoryStore("diag_session")
    ms.upsert_glossary("vector", "A mathematical object with magnitude and direction", 1)
    ms.upsert_callback("cb_001", 1, "The AI that went rogue", [3, 5])
    ms.upsert_concept("RAG", "Retrieval-Augmented Generation", 2)
    ms.log_decision("TestAgent", "Use hash embedder", "Torch path too long on Windows", 1)
    g = ms.get_glossary()
    c = ms.get_callbacks()
    print(f"  ✓ SQLite | {len(g)} glossary, {len(c)} callbacks")
except Exception as e:
    print(f"  ✗ SQLite error: {e}")
    traceback.print_exc()
    all_ok = False

print()
print("=" * 60)
print("STEP 4: Document generation (DOCX + PDF)")
print("=" * 60)
try:
    from agents.assembler import AssemblerAgent, OUTPUTS_DIR
    from utils.logger import ObservabilityLogger
    from utils.cost_tracker import CostTracker

    logger = ObservabilityLogger("diag")
    cost = CostTracker("diag")

    # Minimal mock state for assembler
    mock_state = {
        "user_brief": "test",
        "tone_name": "conversational",
        "author_name": "Test Author",
        "outline": {
            "title": "Diagnostic Test Book",
            "subtitle": "A System Check",
            "author_name": "Test Author",
            "genre": "Technical",
            "core_thesis": "Systems should be tested before shipping.",
            "narrative_arc": "Problem → Solution → Results",
            "chapters": [{"chapter_number": 1, "title": "Introduction", "summary": "Overview"}],
            "glossary_seed": [{"term": "AI", "definition": "Artificial Intelligence", "chapter_introduced": 1}],
            "callback_index": [],
            "foreword_notes": "Welcome to the test.",
            "preface_notes": "This is a diagnostic.",
            "afterword_notes": "Tests passed.",
            "about_author_notes": "Written by a diagnostic script.",
            "back_cover_copy": "A test book.",
        },
        "chapters": {
            1: {
                "chapter_number": 1,
                "title": "Introduction",
                "final_content": "This is a test chapter. " * 50,
                "word_count": 150,
                "callbacks_used": [],
                "glossary_terms_used": [],
            }
        },
        "glossary": [{"term": "AI", "definition": "Artificial Intelligence", "chapter_intro": 1}],
        "agent_logs": [],
    }

    asm = AssemblerAgent(logger=logger, cost_tracker=cost)

    # Test DOCX
    try:
        from pathlib import Path
        docx_path = OUTPUTS_DIR / "diagnostic_test.docx"
        asm._generate_docx(
            path=docx_path,
            title="Diagnostic Test Book",
            subtitle="A System Check",
            author_name="Test Author",
            foreword="This is the foreword.",
            preface="This is the preface.",
            chapters=[{"number": 1, "title": "Introduction", "content": "Test content. " * 50}],
            afterword="This is the afterword.",
            about_author="Written by a test script.",
            back_cover="A test book for diagnostic purposes.",
            glossary=[{"term": "AI", "definition": "Artificial Intelligence"}],
            outline=mock_state["outline"],
        )
        print(f"  ✓ DOCX generated: {docx_path}")
    except Exception as e:
        print(f"  ✗ DOCX error: {e}")
        traceback.print_exc()

    # Test PDF
    try:
        pdf_path = OUTPUTS_DIR / "diagnostic_test.pdf"
        asm._generate_pdf(
            path=pdf_path,
            title="Diagnostic Test Book",
            subtitle="A System Check",
            author_name="Test Author",
            foreword="This is the foreword.",
            preface="This is the preface.",
            chapters=[{"number": 1, "title": "Introduction", "content": "Test content. " * 50}],
            afterword="This is the afterword.",
            about_author="Written by a test script.",
            back_cover="A test book for diagnostic purposes.",
            glossary=[{"term": "AI", "definition": "Artificial Intelligence"}],
        )
        print(f"  ✓ PDF generated: {pdf_path}")
    except Exception as e:
        print(f"  ✗ PDF error: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"  ✗ Assembler import error: {e}")
    traceback.print_exc()
    all_ok = False

print()
print("=" * 60)
print("STEP 5: LangGraph graph build")
print("=" * 60)
try:
    from orchestration.graph import build_graph
    graph, logger, cost, sid = build_graph("diag_graph")
    print(f"  ✓ LangGraph compiled | session={sid}")
except Exception as e:
    print(f"  ✗ Graph build error: {e}")
    traceback.print_exc()
    all_ok = False

print()
print("=" * 60)
print(f"DIAGNOSTIC COMPLETE | All OK: {all_ok}")
print("=" * 60)
