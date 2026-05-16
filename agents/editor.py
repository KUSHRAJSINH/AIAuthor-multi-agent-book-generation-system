"""
EditorAgent — checks consistency, pacing, readability, grammar, and tone alignment.
"""
from __future__ import annotations

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
        temperature=0.4,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class EditorAgent:
    """
    Applies developmental editing to humanized chapter prose.
    Checks pacing, tone consistency, grammar, and readability.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost = cost_tracker
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        tone = state.get("tone_name", "conversational")
        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}
        outline = state.get("outline", {})
        chapters_plans = {p["chapter_number"]: p for p in outline.get("chapters", [])}
        tone_fingerprint = state.get("tone_fingerprint", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append(f"[EditorAgent] Editing {len(chapters)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.editor(tone)

        fingerprint_str = str(tone_fingerprint) if tone_fingerprint else "No fingerprint yet."

        for ch_num, content in chapters.items():
            t0 = self.logger.agent_start("EditorAgent", chapter=int(ch_num))

            humanized = content.get("humanized_content", content.get("raw_content", ""))
            plan = chapters_plans.get(int(ch_num), {})

            user_prompt = user_tmpl.format(
                humanized_prose=humanized,
                chapter_number=ch_num,
                tone=tone,
                chapter_plan_json=str(plan)[:800],
                tone_fingerprint=fingerprint_str[:400],
            )

            self.logger.prompt_log("EditorAgent", system_tmpl, user_prompt, chapter=int(ch_num))

            edited = self._invoke_with_retry(system_tmpl, user_prompt, int(ch_num), humanized)

            content["edited_content"] = edited
            content["word_count"] = len(edited.split())
            chapters[ch_num] = content

            self.logger.agent_end("EditorAgent", t0, chapter=int(ch_num),
                                  tokens=len(edited.split()) * 2)

        state["chapters"] = chapters
        logs = state.get("agent_logs", [])
        logs.append(f"[EditorAgent] ✓ Editing complete")
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
                self.logger.error("EditorAgent", f"Attempt {attempt+1}: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return fallback
                time.sleep(2 ** attempt)
        return fallback
