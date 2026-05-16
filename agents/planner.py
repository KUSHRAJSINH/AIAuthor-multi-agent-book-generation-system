"""
PlannerAgent — generates book outline, chapter structure, narrative arc,
glossary seed, callback index, and tone blueprint.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookOutline, BookState
from prompts.templates import PromptTemplates
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class PlannerAgent:
    """
    Generates a structured BookOutline from a user brief.
    Output is validated against the BookOutline Pydantic schema.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker):
        self.logger = logger
        self.cost = cost_tracker
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        tone = state.get("tone_name", "conversational")
        user_brief = state.get("user_brief", "")
        chapter_count = state.get("chapter_count", 5)
        chapter_length = state.get("chapter_length", 800)
        author_name = state.get("author_name", "AIuthor")

        logs = state.get("agent_logs", [])
        logs.append("[PlannerAgent] Starting book outline generation...")
        state["agent_logs"] = logs

        t0 = self.logger.agent_start("PlannerAgent")

        system_tmpl, user_tmpl = PromptTemplates.planner(tone)
        user_prompt = user_tmpl.format(
            user_brief=user_brief,
            tone=tone,
            chapter_count=chapter_count,
            chapter_length=chapter_length,
            author_name=author_name,
        )

        self.logger.prompt_log("PlannerAgent", system_tmpl, user_prompt)

        outline_dict = self._invoke_with_retry(system_tmpl, user_prompt, tone, chapter_count)

        # Validate with Pydantic
        try:
            outline_obj = BookOutline.model_validate(outline_dict)
            state["outline"] = outline_obj.model_dump()
        except Exception as e:
            self.logger.error("PlannerAgent", f"Schema validation failed: {e}")
            # Use raw dict as fallback
            state["outline"] = outline_dict

        self.logger.agent_end("PlannerAgent", t0)
        logs = state.get("agent_logs", [])
        logs.append(f"[PlannerAgent] ✓ Outline created: '{outline_dict.get('title', 'Untitled')}'")
        state["agent_logs"] = logs
        state["current_chapter_index"] = 0

        return state

    def _invoke_with_retry(self, system: str, user: str, tone: str, chapter_count: int,
                           max_retries: int = 3) -> Dict[str, Any]:
        """Invoke LLM with retry and JSON extraction logic."""
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ]
                response = self.llm.invoke(messages)
                raw = response.content.strip()

                # Extract JSON from response (strip markdown fences if present)
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                # Find first { to last } 
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]

                parsed = json.loads(raw)

                # Ensure required chapters count matches request
                if "chapters" in parsed:
                    chapters = parsed["chapters"]
                    if len(chapters) < chapter_count:
                        # Extend chapters list to match requested count
                        for i in range(len(chapters) + 1, chapter_count + 1):
                            chapters.append({
                                "chapter_number": i,
                                "title": f"Chapter {i}",
                                "summary": f"Continuation of the narrative arc — chapter {i}.",
                                "key_concepts": [],
                                "learning_objectives": [],
                                "callbacks_to_introduce": [],
                                "callbacks_to_resolve": [],
                                "estimated_word_count": 800,
                                "tone_notes": tone,
                            })
                        parsed["chapters"] = chapters

                # Ensure tone field exists
                if "tone" not in parsed or not isinstance(parsed.get("tone"), dict):
                    from prompts.templates import TONE_DIRECTIVES
                    parsed["tone"] = {
                        "name": tone,
                        "system_instruction": TONE_DIRECTIVES.get(tone, ""),
                        "vocabulary_level": "intermediate",
                        "sentence_variety": "mixed",
                        "person": "second",
                        "formality": "casual",
                        "emotional_temperature": "warm",
                        "example_opening": "",
                    }

                return parsed

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error("PlannerAgent", f"Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    return self._fallback_outline(tone, chapter_count)
                time.sleep(2 ** attempt)

        return self._fallback_outline(tone, chapter_count)

    def _fallback_outline(self, tone: str, chapter_count: int) -> Dict[str, Any]:
        """Minimal valid outline if LLM fails repeatedly."""
        from prompts.templates import TONE_DIRECTIVES
        chapters = []
        for i in range(1, chapter_count + 1):
            chapters.append({
                "chapter_number": i,
                "title": f"Chapter {i}: The Journey Continues",
                "summary": f"Chapter {i} explores the next stage of the narrative.",
                "key_concepts": ["concept A", "concept B"],
                "learning_objectives": ["understand the core idea", "apply the lesson"],
                "callbacks_to_introduce": [f"cb_{i:03d}"],
                "callbacks_to_resolve": [],
                "estimated_word_count": 800,
                "tone_notes": tone,
            })
        return {
            "title": "A Book Worth Reading",
            "subtitle": "A Journey Through Ideas",
            "author_name": "AIuthor",
            "genre": "Non-fiction",
            "target_audience": "General readers",
            "core_thesis": "Ideas shape the world.",
            "narrative_arc": "Introduction → Development → Resolution",
            "chapters": chapters,
            "glossary_seed": [],
            "callback_index": [],
            "tone": {
                "name": tone,
                "system_instruction": TONE_DIRECTIVES.get(tone, ""),
                "vocabulary_level": "intermediate",
                "sentence_variety": "mixed",
                "person": "second",
                "formality": "casual",
                "emotional_temperature": "warm",
                "example_opening": "",
            },
            "foreword_notes": "",
            "preface_notes": "",
            "afterword_notes": "",
            "about_author_notes": "",
            "back_cover_copy": "",
        }
