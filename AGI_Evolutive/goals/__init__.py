"""High level goal management system."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional

from .curiosity import CuriosityEngine
from .dag_store import DagStore, GoalNode
from .heuristics import HeuristicRegistry, default_heuristics
from .intention_classifier import IntentionModel


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

    def __init__(
        self,
        architecture=None,
        memory=None,
        reasoning=None,
        *,
        persist_path: str = "data/goals.json",
        dashboard_path: str = "data/goals_dashboard.json",
        intention_data_path: str = "data/goal_intentions.json",
    ):
        self.architecture = architecture
        self.memory = memory
        self.reasoning = reasoning

        self.store = DagStore(
            persist_path=persist_path,
            dashboard_path=dashboard_path,
        )
        self.metadata: Dict[str, GoalMetadata] = {}
        self.pending_actions: Deque[Dict[str, Any]] = deque()
        self.curiosity = CuriosityEngine(architecture=architecture)
        self.heuristics: HeuristicRegistry = default_heuristics()
        self.intention_model = IntentionModel(data_path=intention_data_path)
        self.intent_confidence_threshold = 0.55

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
            description="ÉVOLUER",
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
        if node.created_by == "curiosity":
            self.curiosity.observe_goal_feedback(node.id, user_msg)

    def _should_autopropose(self) -> bool:
        return (time.time() - self.last_auto_proposal_at) >= self.auto_proposal_interval

    def _propose_curiosity_goals(self) -> None:
        active = self.store.get_active()
        parent_payload = active.to_dict() if active else None
        proposals = self.curiosity.suggest_subgoals(parent_payload)
        for proposal in proposals:
            node = self.store.add_goal(**proposal)
            if node.created_by == "curiosity":
                self.curiosity.register_proposal(node.id, proposal)
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
        heuristic_actions = self.heuristics.match(goal)
        if heuristic_actions:
            return heuristic_actions

        meta = self.metadata.get(goal.id)
        prediction = self.intention_model.predict(goal, meta)
        if prediction.label and prediction.confidence >= self.intent_confidence_threshold:
            classified = self._actions_from_intention(goal, meta, prediction.label, prediction.confidence)
            if classified:
                return classified
        elif prediction.confidence > 0.0:
            exploratory = self._probe_actions(goal, meta, prediction.confidence)
            if exploratory:
                return exploratory

        return self._default_actions(goal, meta)

    def _default_actions(
        self, goal: GoalNode, metadata: Optional[GoalMetadata]
    ) -> Deque[Dict[str, Any]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
        }
        if metadata:
            payload_base.update(metadata.to_payload())

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

    def _actions_from_intention(
        self,
        goal: GoalNode,
        metadata: Optional[GoalMetadata],
        label: str,
        confidence: float,
    ) -> Optional[Deque[Dict[str, Any]]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
            "confidence": confidence,
        }
        if metadata:
            payload_base.update(metadata.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        priority = goal.priority or 0.5
        label = label.lower()
        if label in {"plan", "planning"}:
            actions.append(
                {
                    "type": "plan",
                    "payload": {**payload_base, "child_ids": list(goal.child_ids)},
                    "priority": max(0.1, priority),
                }
            )
            if not goal.child_ids:
                actions.append(
                    {
                        "type": "learn_concept",
                        "payload": {**payload_base, "hint": "identifier des sous-objectifs"},
                        "priority": min(1.0, priority + 0.05),
                    }
                )
        elif label in {"reflect", "analyse"}:
            actions.append(
                {
                    "type": "reflect",
                    "payload": {**payload_base, "hint": "approfondir le contexte"},
                    "priority": max(0.1, priority),
                }
            )
        elif label in {"learn_concept", "research", "explore"}:
            actions.append(
                {
                    "type": "learn_concept",
                    "payload": {**payload_base, "hint": "recherche ciblée"},
                    "priority": min(1.0, priority + 0.1),
                }
            )
        elif label in {"execute", "act"}:
            actions.append(
                {
                    "type": "execute_goal",
                    "payload": {**payload_base, "hint": "passer à l'action"},
                    "priority": min(1.0, priority + 0.05),
                }
            )
        else:
            return None

        return actions if actions else None

    def _probe_actions(
        self, goal: GoalNode, metadata: Optional[GoalMetadata], confidence: float
    ) -> Deque[Dict[str, Any]]:
        payload_base = {
            "goal_id": goal.id,
            "description": goal.description,
            "criteria": list(goal.criteria),
            "confidence": confidence,
        }
        if metadata:
            payload_base.update(metadata.to_payload())

        actions: Deque[Dict[str, Any]] = deque()
        actions.append(
            {
                "type": "probe_goal",
                "payload": {**payload_base, "hint": "clarifier l'intention ou demander des précisions"},
                "priority": max(0.1, goal.priority * 0.95),
            }
        )
        actions.append(
            {
                "type": "reflect",
                "payload": {**payload_base, "hint": "collecter des indices complémentaires"},
                "priority": goal.priority,
            }
        )
        return actions

    def record_goal_outcome(
        self,
        goal_id: str,
        *,
        succeeded: bool,
        executed_actions: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        if not succeeded or not executed_actions:
            return
        goal = self.store.get_goal(goal_id)
        if not goal:
            return
        for action in executed_actions:
            action_type = action.get("type") if isinstance(action, dict) else None
            if not action_type:
                continue
            metadata = self.metadata.get(goal_id)
            self.intention_model.update(action_type, goal, metadata)
            break

    # ------------------------------------------------------------------
    def refresh_plans(self) -> None:
        """Compatibility stub for the scheduler."""
        self._ensure_pending_actions()
