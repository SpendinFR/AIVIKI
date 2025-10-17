"""Global workspace coordination utilities."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from AGI_Evolutive.core.structures.mai import Bid


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
        self._pending_bids: List[Bid] = []
        self._trace_last_winners: List[Bid] = []

    def broadcast(self, item: Any) -> None:
        self.broadcasts.append(item)
        if len(self.broadcasts) > 1000:
            self.broadcasts.pop(0)

    def submit_bid(self, channel: str, bid_type: str, attention: float, payload: Dict[str, Any]) -> None:
        """Register a bid emitted by a module."""
        attention = max(0.0, min(1.0, attention))
        self._bids.append((channel, bid_type, attention, payload))

    def _policy_score_safe(self, frame: Any, option: Optional[str] = None) -> float:
        """Return a policy score using the best available API or a RAG-based fallback."""

        pol = getattr(self, "policy", None)
        try:
            if pol and hasattr(pol, "evaluate"):
                return float(pol.evaluate(frame, option=option))
            if pol and hasattr(pol, "confidence_for"):
                return float(pol.confidence_for(frame, option=option))
            if pol and hasattr(pol, "confidence"):
                return float(pol.confidence(frame))
        except Exception:
            pass

        if option == "no_rag_use":
            return 0.0

        signals = getattr(frame, "signals", {}) or {}
        top1 = float(signals.get("rag_top1", 0.0))
        mean = float(signals.get("rag_mean", 0.0))
        div = float(signals.get("rag_diversity", 0.0))
        n_docs = float(signals.get("rag_docs", 0.0))
        fallback = 0.45 * top1 + 0.35 * mean + 0.15 * div + 0.05 * min(n_docs / 5.0, 1.0)
        return max(0.0, min(1.0, fallback))

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

    # ------------------------------------------------------------------
    # MAI integration helpers
    def submit(self, bid: Bid) -> None:
        self._pending_bids.append(bid)

    def step(self, state: Dict, timebox_iters: int = 2) -> None:
        groups: Dict[str, List[Bid]] = {}
        now = time.time()
        self._pending_bids = [
            b for b in self._pending_bids if (b.expires_at or now + 1e9) > now
        ]
        for bid in self._pending_bids:
            groups.setdefault(bid.action_hint, []).append(bid)

        scored: List[Tuple[float, Bid]] = []
        for hint, bids in groups.items():
            avg_gain = sum(max(0.0, x.expected_info_gain) for x in bids) / max(1, len(bids))
            avg_urg = sum(max(0.0, x.urgency) for x in bids) / max(1, len(bids))
            avg_aff = sum(max(0.0, x.affect_value) for x in bids) / max(1, len(bids))
            avg_cost = sum(max(0.0, x.cost) for x in bids) / max(1, len(bids))
            score = (avg_gain + 0.5 * avg_urg + 0.2 * avg_aff) - 0.3 * avg_cost
            scored.append((score, bids[0]))

        scored.sort(key=lambda t: t[0], reverse=True)
        K = min(5, len(scored))
        self._trace_last_winners = [bid for _, bid in scored[:K]]

    def winners(self) -> List[Bid]:
        return list(self._trace_last_winners)

    def last_trace(self) -> List[Bid]:
        return list(self._trace_last_winners)

    def process_frame(self, frame: Any) -> None:
        """Run planner/policy pipeline for a frame and publish RAG evidence bids."""
        if self.planner is not None:
            self.planner.plan(frame)
        if self.policy is None:
            return

        current_u = self._policy_score_safe(frame, option="no_rag_use")
        rag_u = self._policy_score_safe(frame, option="use_rag")
        utility_delta = rag_u - current_u
        self._submit_rag_bid(frame, utility_delta)
