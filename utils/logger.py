"""
Observability logger — writes structured traces, prompt logs, memory events,
and timing data to logs/ directory.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservabilityLogger:
    """Centralised logger. Each run writes to a session file."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_file = LOG_DIR / f"session_{session_id}.jsonl"
        self._start = time.perf_counter()

    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("session_id", self.session_id)
        record.setdefault("timestamp", _timestamp())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---- Public API --------------------------------------------------------

    def agent_start(self, agent: str, chapter: int = 0) -> float:
        t = time.perf_counter()
        self._write({"event": "agent_start", "agent": agent, "chapter": chapter})
        return t

    def agent_end(self, agent: str, start_time: float, chapter: int = 0, tokens: int = 0) -> None:
        elapsed = round(time.perf_counter() - start_time, 3)
        self._write({
            "event": "agent_end",
            "agent": agent,
            "chapter": chapter,
            "elapsed_seconds": elapsed,
            "tokens_used": tokens,
        })

    def prompt_log(self, agent: str, system: str, user: str, chapter: int = 0) -> None:
        self._write({
            "event": "prompt",
            "agent": agent,
            "chapter": chapter,
            "system_prompt": system[:500],   # truncate for storage
            "user_prompt": user[:500],
        })

    def memory_read(self, store: str, key: str, result_count: int = 0) -> None:
        self._write({
            "event": "memory_read",
            "store": store,
            "key": key,
            "result_count": result_count,
        })

    def memory_write(self, store: str, key: str, value_preview: str = "") -> None:
        self._write({
            "event": "memory_write",
            "store": store,
            "key": key,
            "value_preview": str(value_preview)[:200],
        })

    def error(self, agent: str, message: str, chapter: int = 0) -> None:
        self._write({
            "event": "error",
            "agent": agent,
            "chapter": chapter,
            "message": message,
        })

    def info(self, agent: str, message: str, chapter: int = 0) -> None:
        self._write({
            "event": "info",
            "agent": agent,
            "chapter": chapter,
            "message": message,
        })

    def get_log_path(self) -> str:
        return str(self.log_file)

    def read_log(self) -> list[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
