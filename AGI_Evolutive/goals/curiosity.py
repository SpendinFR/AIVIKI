"""Goal generation heuristics driven by curiosity signals."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class CuriosityEngine:
    """Generate candidate sub-goals based on simple heuristics."""

    def __init__(self, architecture=None):
        self.architecture = architecture

    def suggest_subgoals(
        self,
        parent_goal: Optional[Dict[str, Any]] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` sub-goal dictionaries for :class:`DagStore`."""

        context = self._collect_context()
        gaps = self._identify_gaps(context)
        proposals = [self._gap_to_goal(gap, parent_goal) for gap in gaps]

        if len(proposals) < k:
            proposals.append(
                {
                    "description": "Explorer un concept peu maîtrisé et produire une synthèse.",
                    "criteria": ["Fournir un résumé structuré et une auto-évaluation."],
                    "created_by": "curiosity",
                    "value": 0.55,
                    "competence": 0.5,
                    "curiosity": 0.8,
                    "urgency": 0.35,
                    "parent_ids": [parent_goal["id"]] if parent_goal else [],
                }
            )

        random.shuffle(proposals)
        return proposals[:k]

    # ------------------------------------------------------------------
    def _collect_context(self) -> Dict[str, Any]:
        metacog = getattr(self.architecture, "metacognition", None)
        reasoning = getattr(self.architecture, "reasoning", None)

        status: Dict[str, Any] = {}
        if metacog and hasattr(metacog, "get_metacognitive_status"):
            try:
                status = metacog.get_metacognitive_status()
            except Exception:
                status = {}

        reasoning_stats: Dict[str, Any] = {}
        if reasoning and hasattr(reasoning, "get_reasoning_stats"):
            try:
                reasoning_stats = reasoning.get_reasoning_stats()
            except Exception:
                reasoning_stats = {}

        return {"metacog": status, "reasoning": reasoning_stats}

    def _identify_gaps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        performance = context.get("metacog", {}).get("performance_metrics", {})
        low_metrics = sorted(
            (item for item in performance.items() if item[1] < 0.45),
            key=lambda item: item[1],
        )

        gaps = [
            {"domain": name, "score": value, "severity": float(1.0 - value)}
            for name, value in low_metrics[:3]
        ]

        reasoning_errors = context.get("reasoning", {}).get("common_errors", [])
        gaps.extend(
            {"domain": "reasoning_error", "score": 0.3, "hint": err, "severity": 0.6}
            for err in reasoning_errors[:2]
        )

        if not gaps:
            gaps.append({"domain": "exploration", "score": 0.5, "severity": 0.4})

        return gaps

    def _gap_to_goal(self, gap: Dict[str, Any], parent_goal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        description = self._describe_gap(gap)
        criteria = self._default_criteria(gap)
        parent_ids = [parent_goal["id"]] if parent_goal and "id" in parent_goal else []

        base_value = 0.5 + 0.2 * random.random()
        competence = max(0.3, min(0.7, 0.45 + (random.random() - 0.5) * 0.3))
        curiosity = min(1.0, 0.6 + 0.3 * random.random())
        urgency = 0.3 + 0.2 * random.random()

        return {
            "description": description,
            "criteria": criteria,
            "created_by": "curiosity",
            "value": base_value,
            "competence": competence,
            "curiosity": curiosity,
            "urgency": urgency,
            "parent_ids": parent_ids,
        }

    def _describe_gap(self, gap: Dict[str, Any]) -> str:
        domain = gap.get("domain")
        if domain == "reasoning_error":
            return f"Analyser et corriger une erreur de raisonnement: {gap.get('hint', 'non spécifié')}"
        if domain == "exploration":
            return "Explorer un nouveau sujet pour enrichir la base de connaissances."
        return f"Améliorer la métrique {domain} par une expérimentation ciblée."

    def _default_criteria(self, gap: Dict[str, Any]) -> List[str]:
        domain = gap.get("domain")
        if domain == "reasoning_error":
            return ["Documenter 3 contre-exemples et une stratégie de prévention."]
        if domain == "exploration":
            return ["Produire une carte mentale du sujet exploré."]
        return ["Mesurer une amélioration significative après 3 essais contrôlés."]
