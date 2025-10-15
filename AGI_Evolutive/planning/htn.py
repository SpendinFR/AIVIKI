"""HTN planning helpers tied to the belief graph and ontology."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from AGI_Evolutive.reasoning.structures import HTNPlanner as _BasePlanner


class HTNPlanner(_BasePlanner):
    """Small extension that records planning context for calibration."""

    def __init__(self, beliefs: Any, ontology: Any) -> None:
        super().__init__()
        self.beliefs = beliefs
        self.ontology = ontology

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:  # type: ignore[override]
        context = dict(context or {})
        context.setdefault("known_beliefs", len(self.beliefs.query(active_only=True)))
        return super().plan(goal, context=context)

