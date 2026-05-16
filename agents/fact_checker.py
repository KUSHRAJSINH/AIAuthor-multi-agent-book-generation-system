"""
FactCheckerAgent — verifies factual claims against the fact registry,
softens unsupported assertions, prevents hallucinations.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState
from prompts.templates import PromptTemplates
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class FactCheckerAgent:
    """
    Verifies factual claims in edited prose against the fact registry.
    Softens unsupported claims rather than removing them.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost = cost_tracker
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        tone = state.get("tone_name", "conversational")
        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}
        research_packets = state.get("research_packets", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append(f"[FactCheckerAgent] Fact-checking {len(chapters)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.fact_checker(tone)

        # Aggregate fact registry across all chapters
        all_facts: list = []
        for packet in research_packets.values():
            all_facts.extend(packet.get("fact_registry", []))

        for ch_num, content in chapters.items():
            t0 = self.logger.agent_start("FactCheckerAgent", chapter=int(ch_num))

            edited = content.get("edited_content", content.get("humanized_content", content.get("raw_content", "")))

            # Get facts for this specific chapter
            chapter_facts = [f for f in all_facts if f.get("chapter_number", int(ch_num)) == int(ch_num)]
            if not chapter_facts:
                # Use all facts as general reference
                chapter_facts = all_facts[:5]

            user_prompt = user_tmpl.format(
                edited_prose=edited,
                fact_registry_json=json.dumps(chapter_facts, indent=2)[:1500],
                chapter_number=ch_num,
            )

            self.logger.prompt_log("FactCheckerAgent", system_tmpl, user_prompt, chapter=int(ch_num))

            fact_checked = self._invoke_with_retry(system_tmpl, user_prompt, int(ch_num), edited)

            content["fact_checked_content"] = fact_checked
            content["final_content"] = fact_checked
            content["word_count"] = len(fact_checked.split())
            chapters[ch_num] = content

            self.logger.agent_end("FactCheckerAgent", t0, chapter=int(ch_num))

        state["chapters"] = chapters
        logs = state.get("agent_logs", [])
        logs.append(f"[FactCheckerAgent] ✓ Fact-checking complete")
        state["agent_logs"] = logs

        return state

    def _invoke_with_retry(self, system: str, user: str, chapter_num: int,
                           fallback: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                messages = [SystemMessage(content=system), HumanMessage(content=user)]
                response = self.llm.invoke(messages)
                result = response.content.strip()
                if len(result) > 100:
                    return result
                raise ValueError("Response too short")
            except Exception as e:
                self.logger.error("FactCheckerAgent", f"Attempt {attempt+1}: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return fallback
                time.sleep(2 ** attempt)
        return fallback
