"""
Evaluation pipeline — LLM-as-judge automated evaluation with rubric scoring.
Produces structured evaluation reports and writes them to evaluations/ directory.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState, EvaluationReport, EvaluationMetric
from prompts.templates import PromptTemplates
from utils.logger import ObservabilityLogger

EVAL_DIR = Path("evaluations")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


class EvaluationPipeline:
    """
    Runs automated evaluation after book generation is complete.
    Uses LLM-as-judge with a 6-metric rubric.
    """

    def __init__(self, logger: ObservabilityLogger):
        self.logger = logger
        self.llm = _get_llm()

    def run(self, state: BookState) -> BookState:
        outline = state.get("outline", {})
        tone = state.get("tone_name", "conversational")
        chapters: Dict[int, Dict[str, Any]] = state.get("chapters", {}) or {}
        callback_index = state.get("callback_index", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append("[EvaluationPipeline] Running automated evaluation...")
        state["agent_logs"] = logs

        # Build chapter excerpts for evaluation
        excerpts = []
        for ch_num in sorted(chapters.keys()):
            content = chapters[ch_num]
            final = content.get("final_content",
                    content.get("fact_checked_content",
                    content.get("edited_content",
                    content.get("raw_content", ""))))
            excerpts.append(f"--- Chapter {ch_num}: {content.get('title', '')} ---\n{final[:2000]}")

        chapter_excerpts_str = "\n\n".join(excerpts[:3])  # Limit to first 3 chapters for context

        system_tmpl, user_tmpl = PromptTemplates.evaluator()
        user_prompt = user_tmpl.format(
            title=outline.get("title", "Untitled"),
            tone=tone,
            chapter_excerpts=chapter_excerpts_str,
            callback_index=json.dumps(callback_index, indent=2)[:500],
        )

        raw_eval = self._invoke_with_retry(system_tmpl, user_prompt)

        # Parse and validate evaluation
        report = self._build_report(raw_eval, outline, tone, len(chapters))

        # Save to evaluations/
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = EVAL_DIR / f"eval_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        state["evaluation_report"] = report.model_dump()

        logs = state.get("agent_logs", [])
        logs.append(f"[EvaluationPipeline] ✓ Evaluation complete — Overall: {report.overall_score}/10")
        state["agent_logs"] = logs

        self.logger.info("EvaluationPipeline",
                         f"Evaluation saved: {report_path} | Score: {report.overall_score}")

        return state

    def _build_report(self, raw: Dict, outline: Dict, tone: str, chapter_count: int) -> EvaluationReport:
        metrics_data = raw.get("metrics", [])
        metrics = []
        for m in metrics_data:
            try:
                metrics.append(EvaluationMetric(
                    metric_name=m.get("metric_name", "unknown"),
                    score=float(m.get("score", 5.0)),
                    rationale=m.get("rationale", ""),
                    evidence=m.get("evidence", []),
                ))
            except Exception:
                pass

        # Ensure all 6 metrics exist
        existing_names = {m.metric_name for m in metrics}
        required = [
            "structural_completeness", "tone_fidelity", "ai_tell_detection",
            "callback_consistency", "fact_grounding", "readability"
        ]
        for req in required:
            if req not in existing_names:
                metrics.append(EvaluationMetric(
                    metric_name=req,
                    score=5.0,
                    rationale="Could not be evaluated — insufficient data",
                    evidence=[],
                ))

        report = EvaluationReport(
            book_title=outline.get("title", "Untitled"),
            tone=tone,
            chapter_count=chapter_count,
            metrics=metrics,
            llm_judge_verdict=raw.get("llm_judge_verdict", ""),
            recommendations=raw.get("recommendations", []),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        report.overall_score = report.compute_overall()
        return report

    def _invoke_with_retry(self, system: str, user: str, max_retries: int = 3) -> Dict:
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

                return json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error("EvaluationPipeline", f"Attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {
                        "metrics": [],
                        "overall_score": 5.0,
                        "llm_judge_verdict": "Evaluation could not be completed due to an error.",
                        "recommendations": ["Retry the evaluation pipeline."],
                    }
                time.sleep(2 ** attempt)
        return {}
