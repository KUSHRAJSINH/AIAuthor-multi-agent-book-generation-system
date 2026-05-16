"""
AIuthor — Streamlit Frontend
"""
from __future__ import annotations

import os
import sys
import time
import json
import threading
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.graph import run_pipeline

st.set_page_config(
    page_title="AIuthor — AI Book Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); min-height: 100vh; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.15));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-score {
    font-size: 2.5rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}
.metric-name {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.log-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #58a6ff;
    max-height: 280px;
    overflow-y: auto;
}

.tone-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.4);
}

.agent-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
    color: #cbd5e1;
}
.agent-step.done { color: #34d399; }
.agent-step.active { color: #60a5fa; }

.download-btn {
    background: linear-gradient(135deg, #059669, #0284c7);
    color: white !important;
    text-decoration: none;
    padding: 0.6rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    display: inline-block;
    margin: 0.3rem;
    transition: all 0.2s;
}

[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "progress" not in st.session_state:
    st.session_state.progress = 0

TONE_COLORS = {
    "conversational": "#34d399",
    "academic": "#60a5fa",
    "storyteller": "#f472b6",
    "motivational": "#fb923c",
    "witty": "#facc15",
}

TONE_DESCRIPTIONS = {
    "conversational": "Warm, direct — like talking to a smart friend",
    "academic": "Measured, scholarly — precise and methodical",
    "storyteller": "Vivid, narrative — scenes and metaphors",
    "motivational": "Energetic, inspiring — propels the reader forward",
    "witty": "Clever, dry humor — intelligence with a knowing smile",
}

AGENT_STEPS = [
    ("🧠", "PlannerAgent", "Generating outline & narrative arc"),
    ("🔍", "ResearcherAgent", "RAG retrieval & fact packets"),
    ("✍️", "WriterAgent", "Writing all chapters"),
    ("✨", "HumanizerAgent", "Removing AI-tells & humanizing prose"),
    ("📝", "EditorAgent", "Consistency, pacing & grammar"),
    ("✅", "FactCheckerAgent", "Verifying claims & softening unsupported facts"),
    ("🧩", "MemoryKeeperAgent", "Persisting memory, callbacks & glossary"),
    ("📦", "AssemblerAgent", "Building DOCX + PDF"),
    ("📊", "EvaluationPipeline", "LLM-as-judge scoring"),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get your key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()

    tone = st.selectbox(
        "🎨 Tone",
        options=list(TONE_COLORS.keys()),
        format_func=lambda x: x.capitalize(),
        index=0,
    )
    color = TONE_COLORS[tone]
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.2);border-left:3px solid {color};'
        f'padding:0.5rem 0.8rem;border-radius:4px;color:{color};font-size:0.85rem;">'
        f'{TONE_DESCRIPTIONS[tone]}</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    chapter_count = st.slider("📚 Chapters", min_value=2, max_value=12, value=5)
    chapter_length = st.slider("📏 Words per Chapter", min_value=400, max_value=2000,
                                value=800, step=100)
    author_name = st.text_input("👤 Author Name", value="AIuthor")

    st.divider()
    st.markdown("### 🤖 Agent Pipeline")
    completed_agents = set()
    if st.session_state.result:
        logs_text = " ".join(st.session_state.logs)
        for _, agent_name, _ in AGENT_STEPS:
            if f"[{agent_name}] ✓" in logs_text or f"[{agent_name}]" in logs_text:
                completed_agents.add(agent_name)

    for icon, agent_name, desc in AGENT_STEPS:
        status = "done" if agent_name in completed_agents else (
            "active" if st.session_state.running else ""
        )
        st.markdown(
            f'<div class="agent-step {status}">{icon} <strong>{agent_name}</strong><br>'
            f'<span style="font-size:0.75rem;color:#64748b;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AIuthor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Production-grade multi-agent AI book generation system</div>',
    unsafe_allow_html=True,
)

tab_generate, tab_output, tab_eval = st.tabs(["📝 Generate", "📖 Output", "📊 Evaluation"])

# ── Tab 1: Generate ───────────────────────────────────────────────────────────
with tab_generate:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Book Brief")
    user_brief = st.text_area(
        "Describe your book",
        placeholder=(
            "Example: A practical guide to building production AI systems for software engineers "
            "who want to move from prototypes to deployed, scalable solutions. Cover LLMs, RAG, "
            "agents, evaluation, and MLOps."
        ),
        height=140,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button(
            "🚀 Generate Book",
            disabled=st.session_state.running,
            use_container_width=True,
        )

    # Progress area
    progress_placeholder = st.empty()
    log_placeholder = st.empty()

    if generate_btn:
        if not groq_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
        elif not user_brief.strip():
            st.error("⚠️ Please enter a book brief.")
        else:
            st.session_state.running = True
            st.session_state.result = None
            st.session_state.logs = []
            st.session_state.progress = 0

            log_lines = []

            def progress_cb(msg: str):
                log_lines.append(msg)
                st.session_state.logs = log_lines.copy()

            with st.spinner("🔄 Pipeline running — this takes a few minutes..."):
                result = run_pipeline(
                    user_brief=user_brief,
                    tone_name=tone,
                    chapter_count=chapter_count,
                    chapter_length=chapter_length,
                    author_name=author_name,
                    progress_callback=progress_cb,
                )

            st.session_state.result = result
            st.session_state.logs = log_lines
            st.session_state.running = False
            st.success("✅ Book generation complete! Check the Output and Evaluation tabs.")
            st.rerun()

    # Show live logs
    if st.session_state.logs:
        log_html = "\n".join(
            f'<div style="color:{"#34d399" if "✓" in l else "#60a5fa" if "[" in l else "#94a3b8"}">{l}</div>'
            for l in st.session_state.logs[-30:]
        )
        log_placeholder.markdown(
            f'<div class="log-box">{log_html}</div>',
            unsafe_allow_html=True,
        )

# ── Tab 2: Output ─────────────────────────────────────────────────────────────
with tab_output:
    result = st.session_state.result
    if not result:
        st.info("Generate a book first using the Generate tab.")
    else:
        outline = result.get("outline", {}) or {}
        chapters = result.get("chapters", {}) or {}

        # Book header
        st.markdown(f"## 📖 {outline.get('title', 'Untitled')}")
        if outline.get("subtitle"):
            st.markdown(f"*{outline['subtitle']}*")
        st.markdown(f"**Author:** {outline.get('author_name', 'AIuthor')} | "
                    f"**Genre:** {outline.get('genre', 'N/A')} | "
                    f"**Chapters:** {len(chapters)} | "
                    f"**Tone:** {result.get('tone_name', 'N/A').capitalize()}")

        st.divider()

        # Downloads
        docx_path = result.get("docx_path")
        pdf_path = result.get("pdf_path")

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if docx_path and Path(docx_path).exists():
                with open(docx_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download DOCX",
                        data=f.read(),
                        file_name=Path(docx_path).name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
            else:
                st.warning("DOCX not available")

        with dcol2:
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=f.read(),
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.warning("PDF not available")

        st.divider()

        # Chapter previews
        st.markdown("### 📚 Chapter Previews")
        for ch_num in sorted(chapters.keys()):
            content = chapters[ch_num]
            final = content.get("final_content",
                    content.get("fact_checked_content",
                    content.get("edited_content",
                    content.get("raw_content", ""))))
            with st.expander(f"Chapter {ch_num}: {content.get('title', '')} "
                             f"({content.get('word_count', len(final.split()))} words)"):
                st.markdown(final[:1500] + ("..." if len(final) > 1500 else ""))

        # Glossary
        glossary = result.get("glossary", []) or []
        if glossary:
            st.divider()
            st.markdown("### 📖 Glossary")
            gcols = st.columns(2)
            for i, entry in enumerate(sorted(glossary, key=lambda x: x.get("term", ""))):
                with gcols[i % 2]:
                    st.markdown(
                        f'<div class="card"><strong style="color:#a78bfa">{entry.get("term","")}</strong>'
                        f'<br><span style="color:#cbd5e1;font-size:0.9rem">{entry.get("definition","")}</span></div>',
                        unsafe_allow_html=True,
                    )

# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    result = st.session_state.result
    if not result or not result.get("evaluation_report"):
        st.info("Evaluation results will appear here after generation.")
    else:
        report = result["evaluation_report"]

        overall = report.get("overall_score", 0)
        color_overall = "#34d399" if overall >= 7 else "#fb923c" if overall >= 5 else "#f87171"

        st.markdown(
            f'<div style="text-align:center;padding:1.5rem;">'
            f'<div style="font-size:0.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:2px;">Overall Score</div>'
            f'<div style="font-size:5rem;font-weight:700;color:{color_overall};line-height:1.1">{overall:.1f}</div>'
            f'<div style="color:#94a3b8;">/ 10.0</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f'**LLM Judge Verdict:** {report.get("llm_judge_verdict", "")}')
        st.divider()

        # Metric cards
        metrics = report.get("metrics", [])
        if metrics:
            st.markdown("### 📊 Metric Breakdown")
            cols = st.columns(3)
            for i, m in enumerate(metrics):
                score = m.get("score", 0)
                mc = "#34d399" if score >= 7 else "#fb923c" if score >= 5 else "#f87171"
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-score" style="color:{mc}">{score:.1f}</div>'
                        f'<div class="metric-name">{m.get("metric_name","").replace("_"," ")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Details"):
                        st.write(m.get("rationale", ""))
                        for ev in m.get("evidence", []):
                            st.markdown(f"• {ev}")

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            st.divider()
            st.markdown("### 💡 Recommendations")
            for rec in recs:
                st.markdown(f"• {rec}")

        # Cost summary
        if result.get("agent_logs"):
            cost_lines = [l for l in result["agent_logs"] if "Cost" in l or "Token" in l]
            if cost_lines:
                st.divider()
                st.markdown("### 💰 Cost Summary")
                st.info(cost_lines[-1])
