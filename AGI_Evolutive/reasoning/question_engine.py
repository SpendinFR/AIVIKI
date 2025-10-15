"""Information gain oriented question engine for abductive reasoning."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .abduction import Hypothesis


class QuestionEngine:
    """Rank potential clarification questions by expected information gain."""

    def __init__(self, beliefs: Any, user_model: Any) -> None:
        self.beliefs = beliefs
        self.user_model = user_model

    # ------------------------------------------------------------------
    def best_question(
        self, hypotheses: Iterable["Hypothesis"], observation: str
    ) -> Optional[str]:
        hyps = list(hypotheses)
        if len(hyps) < 2:
            return None
        self.set_hypotheses(hyps)
        scores = [max(1e-6, min(0.999, h.score)) for h in hyps]
        base_entropy = self._entropy(scores)
        candidates: List[Dict[str, Any]] = []
        for hyp in hyps[:3]:
            text = self._craft_question(hyp, observation)
            if not text:
                continue
            posterior = self._posterior_scenarios(hyp, scores)
            expected = sum(p * self._entropy(dist) for p, dist in posterior)
            gain = max(0.0, base_entropy - expected)
            candidates.append({"text": text, "gain": gain, "score": hyp.score})
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["gain"], item["score"]), reverse=True)
        return candidates[0]["text"]

    # ------------------------------------------------------------------
    def _posterior_scenarios(
        self, hyp: "Hypothesis", scores: List[float]
    ) -> List[tuple[float, List[float]]]:
        yes_scores = [min(0.99, s * (1.2 if h is hyp else 0.9)) for s, h in zip(scores, self._hyp_list)]
        no_scores = [min(0.99, s * (0.6 if h is hyp else 1.1)) for s, h in zip(scores, self._hyp_list)]
        yes_norm = self._normalise(yes_scores)
        no_norm = self._normalise(no_scores)
        return [(0.55, yes_norm), (0.45, no_norm)]

    def _craft_question(self, hyp: "Hypothesis", observation: str) -> Optional[str]:
        cue = hyp.label or "cette hypothèse"
        if hyp.causal_support:
            cue += f" (indices: {hyp.causal_support[0]})"
        return f"Quels éléments concrets confirmeraient {cue} dans « {observation[:80]} » ?"

    # ------------------------------------------------------------------
    def _entropy(self, scores: Iterable[float]) -> float:
        probs = self._normalise(scores)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _normalise(self, scores: Iterable[float]) -> List[float]:
        scores = list(scores)
        total = sum(scores) or 1.0
        return [max(1e-6, s / total) for s in scores]

    @property
    def _hyp_list(self) -> List["Hypothesis"]:
        # Helper for typing convenience; updated externally.
        return getattr(self, "_cached_hypotheses", [])

    def set_hypotheses(self, hyps: List["Hypothesis"]) -> None:
        self._cached_hypotheses = hyps

