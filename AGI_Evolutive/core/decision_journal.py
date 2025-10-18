from __future__ import annotations

import time
import uuid
from typing import Dict, Optional


class DecisionJournal:
    """Links triggers -> decision -> action -> outcome; persists in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store
        self._open: Dict[str, Dict] = {}

    def new(self, decision_ctx: Dict) -> str:
        decision_id = f"dec:{uuid.uuid4().hex[:12]}"
        self._open[decision_id] = {
            "decision_id": decision_id,
            "ctx": decision_ctx,
            "ts_start": time.time(),
            "trace_id": None,
            "action": None,
            "expected_score": None,
        }
        return decision_id

    def attach_trace(self, decision_id: str, trace_id: str) -> None:
        if decision_id in self._open:
            self._open[decision_id]["trace_id"] = trace_id

    def commit_action(self, decision_id: str, action: Dict, expected_score: float) -> None:
        if decision_id in self._open:
            self._open[decision_id]["action"] = action
            self._open[decision_id]["expected_score"] = float(expected_score)

    def close(self, decision_id: str, obtained_score: float, latency_ms: Optional[float] = None) -> None:
        decision = self._open.get(decision_id)
        if not decision:
            return
        ts_end = time.time()
        if latency_ms is None:
            latency_ms = 1000.0 * (ts_end - decision["ts_start"])
        outcome = {
            "kind": "decision",
            "decision_id": decision_id,
            "trigger": decision["ctx"].get("trigger"),
            "mode": decision["ctx"].get("mode"),
            "action": decision.get("action"),
            "expected_score": decision.get("expected_score", 1.0),
            "obtained_score": float(obtained_score),
            "latency_ms": float(latency_ms),
            "trace_id": decision.get("trace_id"),
            "ts": ts_end,
        }
        if self.memory is not None:
            self.memory.add(outcome)
        del self._open[decision_id]
