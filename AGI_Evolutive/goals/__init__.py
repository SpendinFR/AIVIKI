"""High level goal management system."""

from __future__ import annotations

import time
from collections import deque
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional

from .curiosity import CuriosityEngine
from .dag_store import DagStore, GoalNode


class GoalType(Enum):
    SURVIVAL = "survival"
    GROWTH = "growth"
    EXPLORATION = "exploration"
    MASTERY = "mastery"
    SOCIAL = "social"
    CREATIVE = "creative"
    SELF_ACTUALISATION = "self_actualisation"
    COGNITIVE = "cognitive"


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class GoalMetadata:
    """Supplementary information tracked alongside :class:`GoalNode`."""

    goal_type: GoalType = GoalType.GROWTH
    success_criteria: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "goal_type": self.goal_type.value,
            "success_criteria": list(self.success_criteria),
        }


class GoalSystem:
    """Manages a persistent DAG of goals and exposes action suggestions."""

    def __init__(self, architecture=None, memory=None, reasoning=None):
        self.architecture = architecture
        self.memory = memory
        self.reasoning = reasoning

        self.store = DagStore(
            persist_path="data/goals.json",
            dashboard_path="data/goals_dashboard.json",
        )
        self.metadata: Dict[str, GoalMetadata] = {}
        self.pending_actions: Deque[Dict[str, Any]] = deque()
        self.curiosity = CuriosityEngine(architecture=architecture)

        self.active_goal_id: Optional[str] = self.store.active_goal_id
        self.last_auto_proposal_at = 0.0
        self.auto_proposal_interval = 180.0

        self._ensure_root_goal()

    # ------------------------------------------------------------------
    # Public API
    def step(self, user_msg: Optional[str] = None) -> None:
        """Maintain the goal list and ensure actions are available."""

        self._ensure_root_goal()
        self._refresh_active_goal()

        if user_msg:
            self._record_feedback(user_msg)

        if self._should_autopropose():
            self._propose_curiosity_goals()

        self._ensure_pending_actions()

    def get_next_action(self) -> Optional[Dict[str, Any]]:
        self._ensure_pending_actions()
        if self.pending_actions:
            return self.pending_actions.popleft()
        return None

    def pop_next_action(self) -> Optional[Dict[str, Any]]:
        return self.get_next_action()

    def add_goal(
        self,
        description: str,
        *,
        goal_type: GoalType = GoalType.GROWTH,
        criteria: Optional[Iterable[str]] = None,
        parent_ids: Optional[Iterable[str]] = None,
        value: float = 0.5,
        competence: float = 0.5,
        curiosity: float = 0.2,
        urgency: float = 0.3,
        created_by: str = "system",
    ) -> GoalNode:
        node = self.store.add_goal(
            description=description,
            criteria=list(criteria or []),
            created_by=created_by,
            value=value,
            competence=competence,
            curiosity=curiosity,
            urgency=urgency,
            parent_ids=list(parent_ids or []),
        )
        self.metadata[node.id] = GoalMetadata(goal_type=goal_type, success_criteria=list(criteria or []))
        return node

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_root_goal(self) -> None:
        if self.store.nodes:
            return
        root = self.add_goal(
            description="Maintenir et améliorer les capacités cognitives de l'agent.",
            goal_type=GoalType.GROWTH,
            criteria=["Rapporter des progrès réguliers"],
            created_by="system",
            value=0.8,
            curiosity=0.6,
            urgency=0.4,
        )
        self.store.set_active(root.id)
        self.active_goal_id = root.id

    def _refresh_active_goal(self) -> None:
        active = self.store.get_active()
        if active:
            self.active_goal_id = active.id
            return
        top = self.store.topk(1)
        if top:
            chosen = top[0]
            self.store.set_active(chosen.id)
            self.active_goal_id = chosen.id

    def _record_feedback(self, user_msg: str) -> None:
        node = self.store.get_active()
        if not node:
            return
        node.evidence.append({"t": time.time(), "type": "user_feedback", "content": user_msg})
        node.updated_at = time.time()
        self.store.update_goal(node.id, {"evidence": node.evidence})

    def _should_autopropose(self) -> bool:
        return (time.time() - self.last_auto_proposal_at) >= self.auto_proposal_interval

    def _propose_curiosity_goals(self) -> None:
        active = self.store.get_active()
        parent_payload = active.to_dict() if active else None
        proposals = self.curiosity.suggest_subgoals(parent_payload)
        for proposal in proposals:
            node = self.store.add_goal(**proposal)
            self.metadata[node.id] = GoalMetadata(
                goal_type=GoalType.EXPLORATION,
                success_criteria=list(proposal.get("criteria", [])),
            )
        self.last_auto_proposal_at = time.time()

    def _ensure_pending_actions(self) -> None:
        if self.pending_actions:
            return
        active = self.store.get_active()
        if not active:
            return
        self.pending_actions.extend(self._goal_to_actions(active))

    def _goal_to_actions(self, goal: GoalNode) -> Deque[Dict[str, Any]]:
        # Détection d’un objectif d’apprentissage de concept (générique)
        try:
            desc = (goal.description or "").strip()
            m = re.search(r"Apprendre le concept «\s*(.+?)\s*»", desc)
            if m:
                concept = m.group(1)
                actions = deque()
                actions.append({
                    "type": "learn_concept",
                    "payload": {"concept": concept, "goal_id": goal.id},
                    "priority": min(1.0, goal.priority + 0.2),
                })
                return actions
        except Exception:
            pass

        try:
            desc = (goal.description or "").strip()
            m = re.match(r"Résoudre contradiction «\s*(.+?)\s*,\s*(.+?)\s*»", desc)
            if m:
                subject, relation = m.groups()
                base_priority = getattr(goal, "priority", 0.6)
                actions = deque()
                actions.append(
                    {
                        "type": "abduce",
                        "payload": {"observation": f"contradiction:{subject}:{relation}"},
                        "priority": min(1.0, base_priority + 0.1),
                    }
                )
                actions.append(
                    {
                        "type": "assert_fact",
                        "payload": {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "value": "(à confirmer)",
                            "confidence": 0.6,
                        },
                        "priority": base_priority,
                    }
                )
                return actions
        except Exception:
            pass

        meta = self.metadata.get(goal.id)
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
        }
        if meta:
            payload_base.update(meta.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        actions.append(
            {
                "type": "reflect",
                "payload": {**payload_base, "hint": "analyser l'état du but"},
                "priority": goal.priority,
            }
        )
        if goal.child_ids:
            actions.append(
                {
                    "type": "plan",
                    "payload": {**payload_base, "child_ids": list(goal.child_ids)},
                    "priority": goal.priority * 0.9,
                }
            )
        else:
            actions.append(
                {
                    "type": "learn_concept",
                    "payload": {**payload_base, "hint": "collecter des informations pertinentes"},
                    "priority": min(1.0, goal.priority + 0.1),
                }
            )
        return actions

    # ------------------------------------------------------------------
    def refresh_plans(self) -> None:
        """Compatibility stub for the scheduler."""
        self._ensure_pending_actions()
