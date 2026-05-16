"""
WriterAgent — writes chapter prose with continuity, callbacks, and tone adherence.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState, ChapterContent
from prompts.templates import PromptTemplates
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.8,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class WriterAgent:
    """
    Writes chapter prose for every chapter in the outline.
    Uses research packets, memory context, and callbacks for continuity.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost = cost_tracker
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        outline = state.get("outline", {})
        tone = state.get("tone_name", "conversational")
        chapter_length = state.get("chapter_length", 800)
        research_packets = state.get("research_packets", {}) or {}
        chapters_plans = outline.get("chapters", [])

        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append(f"[WriterAgent] Writing {len(chapters_plans)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.writer(tone)

        # Track concepts established so far for continuity
        established_concepts: List[str] = []

        for plan in chapters_plans:
            ch_num = plan.get("chapter_number", 1)
            t0 = self.logger.agent_start("WriterAgent", chapter=ch_num)

            packet = research_packets.get(ch_num, {})
            key_facts = packet.get("key_facts", [])
            research_facts_str = "\n".join(f"- {f}" for f in key_facts) if key_facts else "No specific facts retrieved."

            # Gather callbacks from outline
            callback_index = outline.get("callback_index", [])
            open_cbs = [
                cb["description"] for cb in callback_index
                if cb.get("introduced_in_chapter", 1) < ch_num and not cb.get("resolved", False)
            ]
            cbs_to_intro = plan.get("callbacks_to_introduce", [])
            cbs_to_resolve = plan.get("callbacks_to_resolve", [])

            # Gather glossary terms for this chapter
            glossary_seed = outline.get("glossary_seed", [])
            chapter_terms = [
                g["term"] for g in glossary_seed
                if g.get("chapter_introduced", 1) <= ch_num
            ]

            user_prompt = user_tmpl.format(
                chapter_number=ch_num,
                chapter_title=plan.get("title", f"Chapter {ch_num}"),
                chapter_plan_json=json.dumps(plan, indent=2),
                research_facts=research_facts_str,
                established_concepts=", ".join(established_concepts[-5:]) if established_concepts else "None yet",
                open_callbacks=", ".join(open_cbs) if open_cbs else "None",
                callbacks_to_introduce=", ".join(cbs_to_intro) if cbs_to_intro else "None",
                callbacks_to_resolve=", ".join(cbs_to_resolve) if cbs_to_resolve else "None",
                glossary_terms=", ".join(chapter_terms) if chapter_terms else "None specified",
                target_words=chapter_length,
                tone=tone,
            )

            self.logger.prompt_log("WriterAgent", system_tmpl, user_prompt, chapter=ch_num)

            prose = self._invoke_with_retry(system_tmpl, user_prompt, ch_num, plan)

            # Build ChapterContent
            content = ChapterContent(
                chapter_number=ch_num,
                title=plan.get("title", f"Chapter {ch_num}"),
                raw_content=prose,
                word_count=len(prose.split()),
                callbacks_used=cbs_to_resolve,
                glossary_terms_used=chapter_terms[:5],
            )
            chapters[ch_num] = content.model_dump()

            # Update established concepts for next chapter
            established_concepts.extend(plan.get("key_concepts", []))

            self.logger.agent_end("WriterAgent", t0, chapter=ch_num,
                                  tokens=len(prose.split()) * 2)

        state["chapters"] = chapters
        logs = state.get("agent_logs", [])
        logs.append(f"[WriterAgent] ✓ All {len(chapters)} chapters written")
        state["agent_logs"] = logs

        return state

    def _invoke_with_retry(self, system: str, user: str, chapter_num: int,
                           plan: Dict, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                messages = [SystemMessage(content=system), HumanMessage(content=user)]
                response = self.llm.invoke(messages)
                prose = response.content.strip()
                if len(prose) > 100:
                    return prose
                raise ValueError("Response too short — likely a refusal or error")
            except Exception as e:
                self.logger.error("WriterAgent", f"Attempt {attempt+1} failed: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return self._fallback_prose(plan)
                time.sleep(2 ** attempt)
        return self._fallback_prose(plan)

    def _fallback_prose(self, plan: Dict) -> str:
        title = plan.get("title", "Chapter")
        summary = plan.get("summary", "This chapter explores key ideas.")
        concepts = ", ".join(plan.get("key_concepts", []))
        return (
            f"This chapter, '{title}', takes us deeper into the core of our journey. "
            f"{summary} "
            f"The concepts we encounter — {concepts} — form the backbone of what follows. "
            f"Every idea here connects to the larger arc, building toward the insights waiting in the chapters ahead."
        )
