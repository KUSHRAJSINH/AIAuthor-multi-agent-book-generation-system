"""
LangGraph orchestration DAG for the AIuthor book generation pipeline.

Flow:
  Planner → Researcher → Writer → Humanizer → Editor → FactChecker → MemoryKeeper → Assembler → Evaluator
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from schemas.models import BookState
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.humanizer import HumanizerAgent
from agents.editor import EditorAgent
from agents.fact_checker import FactCheckerAgent
from agents.memory_keeper import MemoryKeeperAgent
from agents.assembler import AssemblerAgent
from evaluator.pipeline import EvaluationPipeline
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def build_graph(session_id: str | None = None) -> tuple:
    """
    Build and compile the LangGraph StateGraph.
    Returns (compiled_graph, logger, cost_tracker, session_id).
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    logger = ObservabilityLogger(session_id=session_id)
    cost = CostTracker(session_id=session_id)

    # Instantiate agents
    planner = PlannerAgent(logger=logger, cost_tracker=cost)
    researcher = ResearcherAgent(logger=logger, cost_tracker=cost, session_id=session_id)
    writer = WriterAgent(logger=logger, cost_tracker=cost)
    humanizer = HumanizerAgent(logger=logger, cost_tracker=cost)
    editor = EditorAgent(logger=logger, cost_tracker=cost)
    fact_checker = FactCheckerAgent(logger=logger, cost_tracker=cost)
    memory_keeper = MemoryKeeperAgent(logger=logger, cost_tracker=cost, session_id=session_id)
    assembler = AssemblerAgent(logger=logger, cost_tracker=cost)
    evaluator = EvaluationPipeline(logger=logger)

    # Build the graph
    graph = StateGraph(BookState)

    # Register nodes
    graph.add_node("planner", planner.run)
    graph.add_node("researcher", researcher.run)
    graph.add_node("writer", writer.run)
    graph.add_node("humanizer", humanizer.run)
    graph.add_node("editor", editor.run)
    graph.add_node("fact_checker", fact_checker.run)
    graph.add_node("memory_keeper", memory_keeper.run)
    graph.add_node("assembler", assembler.run)
    graph.add_node("evaluator", evaluator.run)

    # Define edges (linear DAG)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "humanizer")
    graph.add_edge("humanizer", "editor")
    graph.add_edge("editor", "fact_checker")
    graph.add_edge("fact_checker", "memory_keeper")
    graph.add_edge("memory_keeper", "assembler")
    graph.add_edge("assembler", "evaluator")
    graph.add_edge("evaluator", END)

    compiled = graph.compile()

    return compiled, logger, cost, session_id


def run_pipeline(
    user_brief: str,
    tone_name: str = "conversational",
    chapter_count: int = 5,
    chapter_length: int = 800,
    author_name: str = "AIuthor",
    session_id: str | None = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run the complete book generation pipeline.

    Args:
        user_brief: The book topic/brief from the user.
        tone_name: One of: conversational|academic|storyteller|motivational|witty
        chapter_count: Number of chapters to generate.
        chapter_length: Target word count per chapter.
        author_name: Author name for the book.
        session_id: Optional session ID for reproducibility.
        progress_callback: Optional callable(message: str) for streaming updates.

    Returns:
        Final BookState dict.
    """
    compiled, logger, cost, sid = build_graph(session_id)

    initial_state: BookState = {
        "user_brief": user_brief,
        "tone_name": tone_name,
        "chapter_count": chapter_count,
        "chapter_length": chapter_length,
        "author_name": author_name,
        "outline": None,
        "research_packets": {},
        "chapters": {},
        "callback_index": None,
        "fact_registry": None,
        "glossary": [],
        "tone_fingerprint": None,
        "docx_path": None,
        "pdf_path": None,
        "agent_logs": [],
        "errors": [],
        "current_chapter_index": 0,
        "evaluation_report": None,
    }

    logger.info("Orchestrator", f"Pipeline start — session {sid}")

    if progress_callback:
        progress_callback(f"Session {sid} — Starting pipeline...")

    # Stream node events
    final_state = initial_state.copy()
    try:
        for event in compiled.stream(initial_state, {"recursion_limit": 50}):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                    logs = node_output.get("agent_logs", [])
                    if logs and progress_callback:
                        progress_callback(logs[-1])
    except Exception as e:
        logger.error("Orchestrator", f"Pipeline error: {e}")
        final_state["errors"] = final_state.get("errors", []) + [str(e)]
        if progress_callback:
            progress_callback(f"⚠ Pipeline error: {e}")

    cost.save()
    logger.info("Orchestrator", f"Pipeline complete — {cost.summary()}")

    if progress_callback:
        progress_callback(f"✓ Pipeline complete — {cost.summary()}")

    return final_state
