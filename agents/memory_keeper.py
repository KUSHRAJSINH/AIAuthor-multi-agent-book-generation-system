"""
MemoryKeeperAgent — persists concepts, glossary, callbacks, tone fingerprint,
and decisions to SQLite across chapter generation.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState
from prompts.templates import PromptTemplates
from memory.sqlite_store import MemoryStore
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class MemoryKeeperAgent:
    """
    Analyzes completed chapter content and persists structured memory entries.
    Supports callback repair, glossary updates, and concept tracking.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker, session_id: str):
        self.logger = logger
        self.cost = cost_tracker
        self.session_id = session_id
        self.llm = _get_llm()
        self.store = MemoryStore(session_id=session_id)

    def run(self, state: BookState) -> BookState:
        tone = state.get("tone_name", "conversational")
        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}
        outline = state.get("outline", {})
        chapters_plans = {p["chapter_number"]: p for p in outline.get("chapters", [])}

        logs = state.get("agent_logs", [])
        logs.append(f"[MemoryKeeperAgent] Persisting memory for {len(chapters)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.memory_keeper(tone)

        # Pre-populate glossary seed from outline
        for entry in outline.get("glossary_seed", []):
            self.store.upsert_glossary(
                term=entry.get("term", ""),
                definition=entry.get("definition", ""),
                chapter_intro=entry.get("chapter_introduced", 0),
                tone_variant=entry.get("tone_variant", ""),
            )
            self.logger.memory_write("glossary", entry.get("term", ""))

        # Pre-populate callbacks from outline
        for cb in outline.get("callback_index", []):
            self.store.upsert_callback(
                callback_id=cb.get("callback_id", ""),
                introduced_in=cb.get("introduced_in_chapter", 0),
                description=cb.get("description", ""),
                referenced_in=cb.get("referenced_in_chapters", []),
            )
            self.logger.memory_write("callback_index", cb.get("callback_id", ""))

        # Also persist facts from research packets
        research_packets = state.get("research_packets", {}) or {}
        for ch_num, packet in research_packets.items():
            for fact in packet.get("fact_registry", []):
                self.store.upsert_fact(
                    fact_id=fact.get("fact_id", f"f_ch{ch_num}_auto"),
                    claim=fact.get("claim", ""),
                    supported=fact.get("supported", False),
                    citations=fact.get("citations", []),
                    chapter=int(ch_num),
                    softened=fact.get("softened_claim"),
                )

        combined_tone: Dict = {}

        for ch_num, content in chapters.items():
            t0 = self.logger.agent_start("MemoryKeeperAgent", chapter=int(ch_num))

            final_prose = content.get("final_content", content.get("fact_checked_content",
                             content.get("edited_content", content.get("raw_content", ""))))
            plan = chapters_plans.get(int(ch_num), {})

            existing_callbacks = self.store.get_callbacks()
            existing_glossary = self.store.get_glossary()

            user_prompt = user_tmpl.format(
                final_prose=final_prose[:2000],  # truncate for LLM context
                chapter_number=ch_num,
                chapter_plan_json=json.dumps(plan, indent=2)[:500],
                existing_callbacks_json=json.dumps(existing_callbacks[:10], indent=2)[:500],
                existing_glossary_json=json.dumps(existing_glossary[:10], indent=2)[:500],
            )

            self.logger.prompt_log("MemoryKeeperAgent", system_tmpl, user_prompt, chapter=int(ch_num))

            memory_updates = self._invoke_with_retry(system_tmpl, user_prompt, int(ch_num))

            # Persist extracted memory
            self._persist_updates(memory_updates, int(ch_num))

            # Repair callbacks for this chapter
            self.store.repair_callbacks_for_chapter(int(ch_num))

            # Accumulate tone observations
            tone_obs = memory_updates.get("tone_observations", {})
            if tone_obs:
                combined_tone = self._merge_tone(combined_tone, tone_obs)

            self.logger.agent_end("MemoryKeeperAgent", t0, chapter=int(ch_num))

        # Save final tone fingerprint
        if combined_tone:
            self.store.save_tone_fingerprint(tone_name=tone, data=combined_tone)
            state["tone_fingerprint"] = combined_tone

        # Update state with current memory
        state["glossary"] = self.store.get_glossary()
        state["callback_index"] = {"callbacks": self.store.get_callbacks()}

        logs = state.get("agent_logs", [])
        logs.append(f"[MemoryKeeperAgent] ✓ Memory persisted — "
                    f"{len(state['glossary'])} glossary terms, "
                    f"{len(self.store.get_callbacks())} callbacks")
        state["agent_logs"] = logs

        return state

    def _persist_updates(self, updates: Dict, chapter_num: int) -> None:
        for concept in updates.get("new_concepts", []):
            self.store.upsert_concept(
                concept=concept.get("concept", ""),
                explanation=concept.get("explanation", ""),
                chapter_introduced=chapter_num,
                complexity=concept.get("complexity", "intermediate"),
            )
            self.logger.memory_write("concept_bible", concept.get("concept", ""))

        for gterm in updates.get("new_glossary_terms", []):
            self.store.upsert_glossary(
                term=gterm.get("term", ""),
                definition=gterm.get("definition", ""),
                chapter_intro=chapter_num,
            )
            self.logger.memory_write("glossary", gterm.get("term", ""))

        for cb_upd in updates.get("callback_updates", []):
            existing = self.store.get_callbacks()
            existing_ids = {c["callback_id"] for c in existing}
            if cb_upd.get("callback_id") in existing_ids:
                # Update existing callback
                self.store.upsert_callback(
                    callback_id=cb_upd["callback_id"],
                    introduced_in=chapter_num,
                    description="",
                    referenced_in=cb_upd.get("referenced_in_chapters", []),
                    resolved=cb_upd.get("resolved", False),
                )

        for dec in updates.get("decisions", []):
            self.store.log_decision(
                agent="MemoryKeeperAgent",
                decision=dec.get("decision", ""),
                rationale=dec.get("rationale", ""),
                chapter=chapter_num,
            )

    def _merge_tone(self, base: Dict, new: Dict) -> Dict:
        """Running average of tone metrics."""
        if not base:
            return new.copy()
        result = base.copy()
        for key in ["avg_sentence_length", "contraction_ratio", "question_frequency",
                    "exclamation_frequency", "second_person_ratio"]:
            if key in new and key in base:
                result[key] = (base[key] + new[key]) / 2.0
        result["ai_tell_count"] = base.get("ai_tell_count", 0) + new.get("ai_tell_count", 0)
        samples = base.get("sample_sentences", []) + new.get("sample_sentences", [])
        result["sample_sentences"] = samples[:5]
        return result

    def _invoke_with_retry(self, system: str, user: str, chapter_num: int,
                           max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                messages = [SystemMessage(content=system), HumanMessage(content=user)]
                response = self.llm.invoke(messages)
                raw = response.content.strip()

                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]

                parsed = json.loads(raw)
                return parsed
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error("MemoryKeeperAgent", f"Attempt {attempt+1}: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return {"new_concepts": [], "new_glossary_terms": [],
                            "callback_updates": [], "decisions": [], "tone_observations": {}}
                time.sleep(2 ** attempt)
        return {}
