"""Global workspace coordination utilities."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


def _info_gain_from_rag(signals: Dict[str, float]) -> float:
    """Estimate information gain produced by RAG signals."""
    if not signals:
        return 0.0
    top1 = signals.get("rag_top1", 0.0)
    mean = signals.get("rag_mean", 0.0)
    div = signals.get("rag_diversity", 0.0)
    return max(0.0, min(1.0, 0.5 * top1 + 0.3 * mean + 0.2 * div))


def _urgency_from_frame(frame: Any) -> float:
    """Compute an urgency score based on frame metadata."""
    u = 0.5
    try:
        if getattr(frame, "blocking", False):
            u += 0.3
        if getattr(frame, "deadline_ts", None):
            dt = frame.deadline_ts - time.time()
            if dt < 0:
                u += 0.2
            elif dt < 3600:
                u += 0.15
    except Exception:  # pragma: no cover - defensive guard
        pass
    return max(0.0, min(1.0, u))


class GlobalWorkspace:
    """Simple global workspace capable of collecting broadcasts and bids."""

    def __init__(self, policy: Optional[Any] = None, planner: Optional[Any] = None) -> None:
        self.broadcasts: List[Any] = []
        self._bids: List[Tuple[str, str, float, Dict[str, Any]]] = []
        self.policy = policy
        self.planner = planner

    def broadcast(self, item: Any) -> None:
        self.broadcasts.append(item)
        if len(self.broadcasts) > 1000:
            self.broadcasts.pop(0)

    def submit_bid(self, channel: str, bid_type: str, attention: float, payload: Dict[str, Any]) -> None:
        """Register a bid emitted by a module."""
        attention = max(0.0, min(1.0, attention))
        self._bids.append((channel, bid_type, attention, payload))

    def _submit_rag_bid(self, frame: Any, utility_delta: float) -> None:
        signals = getattr(frame, "signals", {}) or {}
        ig = _info_gain_from_rag(signals)
        urg = _urgency_from_frame(frame)
        attention = ig * max(0.0, utility_delta) * urg

        payload = {
            "type": "RAGEvidence",
            "signals": signals,
            "grounded_context": (getattr(frame, "context", {}) or {}).get("grounded_evidence", []),
        }
        self.submit_bid("evidence", "RAGEvidence", attention, payload)

    def process_frame(self, frame: Any) -> None:
        """Run planner/policy pipeline for a frame and publish RAG evidence bids."""
        if self.planner is not None:
            self.planner.plan(frame)
        if self.policy is None:
            return

        current_u = self.policy.evaluate(frame, option="no_rag_use")
        rag_u = self.policy.evaluate(frame, option="use_rag")
        utility_delta = rag_u - current_u
        self._submit_rag_bid(frame, utility_delta)
