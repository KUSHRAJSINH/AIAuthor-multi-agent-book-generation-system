"""
Cost tracker — maintains a running ledger of estimated API costs
based on token usage and model pricing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Groq llama-3.1-8b-instant pricing (USD per 1M tokens, as of 2025)
PRICE_PER_1M_INPUT = 0.05
PRICE_PER_1M_OUTPUT = 0.08


class CostTracker:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ledger: List[Dict] = []
        self._total_input = 0
        self._total_output = 0

    def record(self, agent: str, input_tokens: int, output_tokens: int, chapter: int = 0) -> None:
        input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT
        output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT
        entry = {
            "agent": agent,
            "chapter": chapter,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }
        self.ledger.append(entry)
        self._total_input += input_tokens
        self._total_output += output_tokens

    def total_cost(self) -> float:
        return sum(e["total_cost_usd"] for e in self.ledger)

    def total_tokens(self) -> int:
        return self._total_input + self._total_output

    def save(self) -> str:
        path = LOG_DIR / f"cost_ledger_{self.session_id}.json"
        payload = {
            "session_id": self.session_id,
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_cost_usd": self.total_cost(),
            "entries": self.ledger,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(path)

    def summary(self) -> str:
        return (
            f"Tokens: {self.total_tokens():,} | "
            f"Est. Cost: ${self.total_cost():.4f} USD"
        )
