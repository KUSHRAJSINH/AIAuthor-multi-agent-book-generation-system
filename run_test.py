"""
Live end-to-end pipeline test. Run as:
    python run_test.py YOUR_GROQ_API_KEY
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

# Accept key as argument or env
if len(sys.argv) > 1:
    os.environ["GROQ_API_KEY"] = sys.argv[1]

key = os.environ.get("GROQ_API_KEY", "")
if not key or key == "your_groq_api_key_here":
    print("[ERROR] No GROQ_API_KEY provided.")
    print("Usage: python run_test.py YOUR_GROQ_KEY")
    print("  OR:  set GROQ_API_KEY=YOUR_GROQ_KEY  then  python run_test.py")
    sys.exit(1)

print(f"[INFO] Key loaded: {key[:8]}...")

from orchestration.graph import run_pipeline

print("[INFO] Starting 3-chapter test run...")
print()

result = run_pipeline(
    user_brief=(
        "A practical guide for software engineers who want to build and ship real AI systems — "
        "not just prototype them. Cover the full stack: prompt engineering, RAG pipelines, "
        "multi-agent orchestration, evaluation, and MLOps. Each chapter exposes a real mistake "
        "engineers make and shows how to fix it with concrete patterns."
    ),
    tone_name="conversational",
    chapter_count=3,
    chapter_length=500,
    author_name="Test Author",
    progress_callback=lambda m: print(f"  >> {m}"),
)

print()
print("=" * 55)
print("RESULTS")
print("=" * 55)

outline = result.get("outline") or {}
chapters = result.get("chapters") or {}
errors = result.get("errors") or []

print(f"Book title   : {outline.get('title', 'N/A')}")
print(f"Chapters     : {len(chapters)}")
for num in sorted(chapters.keys()):
    ch = chapters[num]
    content = ch.get("final_content") or ch.get("raw_content") or ""
    wc = len(content.split())
    print(f"  Chapter {num}: {ch.get('title', '')} ({wc} words)")

docx = result.get("docx_path", "NOT GENERATED")
pdf  = result.get("pdf_path", "NOT GENERATED")
print(f"DOCX         : {docx}")
print(f"PDF          : {pdf}")

ev = result.get("evaluation_report") or {}
if ev:
    print(f"Eval score   : {ev.get('overall_score', 0)}/10")
    print(f"Verdict      : {ev.get('llm_judge_verdict', '')[:120]}")

if errors:
    print(f"Errors       : {errors}")
else:
    print("Errors       : None")

print("=" * 55)
