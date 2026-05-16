"""
HumanizerAgent — rewrites AI-like prose to feel authentically human.
Removes AI-tell phrases, varies rhythm, adds emotional cadence.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState
from prompts.templates import PromptTemplates
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker

# AI-tell patterns to detect and flag
AI_TELLS = [
    r"\bit is important to note\b",
    r"\bdelve into\b",
    r"\bdive into\b",
    r"\blandscape of\b",
    r"\bin today's fast.paced world\b",
    r"\bnot only\b.{1,60}\bbut also\b",
    r"\bas we explore\b",
    r"\blet us explore\b",
    r"\bin conclusion\b",
    r"\bto summarize\b",
    r"\bin summary\b",
    r"\bgame.changer\b",
    r"\bparadigm shift\b",
    r"\brevolutionary\b",
    r"\bseamlessly\b",
    r"\bleverage\b",
    r"\bharness\b",
    r"\butilize\b",
    r"\bensure\b",
    r"\bit's worth noting\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bhowever,? it is\b",
]


def _count_ai_tells(text: str) -> int:
    count = 0
    for pattern in AI_TELLS:
        count += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return count


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.75,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class HumanizerAgent:
    """
    Rewrites each chapter's raw prose to remove AI-tells and improve
    human-like rhythm, contractions, and emotional engagement.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost = cost_tracker
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        tone = state.get("tone_name", "conversational")
        chapter_length = state.get("chapter_length", 800)
        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append(f"[HumanizerAgent] Humanizing {len(chapters)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.humanizer(tone)

        for ch_num, content in chapters.items():
            t0 = self.logger.agent_start("HumanizerAgent", chapter=int(ch_num))

            raw_prose = content.get("raw_content", "")
            ai_tell_count_before = _count_ai_tells(raw_prose)

            user_prompt = user_tmpl.format(
                raw_prose=raw_prose,
                chapter_number=ch_num,
                tone=tone,
                target_words=chapter_length,
            )

            self.logger.prompt_log("HumanizerAgent", system_tmpl, user_prompt, chapter=int(ch_num))

            humanized = self._invoke_with_retry(system_tmpl, user_prompt, int(ch_num), raw_prose)
            ai_tell_count_after = _count_ai_tells(humanized)

            content["humanized_content"] = humanized
            content["word_count"] = len(humanized.split())
            chapters[ch_num] = content

            self.logger.agent_end(
                "HumanizerAgent", t0, chapter=int(ch_num),
                tokens=len(humanized.split()) * 2
            )
            self.logger.info(
                "HumanizerAgent",
                f"Ch.{ch_num}: AI-tells reduced from {ai_tell_count_before} → {ai_tell_count_after}",
                chapter=int(ch_num)
            )

        state["chapters"] = chapters
        logs = state.get("agent_logs", [])
        logs.append(f"[HumanizerAgent] ✓ Humanization complete")
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
                self.logger.error("HumanizerAgent", f"Attempt {attempt+1}: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return fallback
                time.sleep(2 ** attempt)
        return fallback
