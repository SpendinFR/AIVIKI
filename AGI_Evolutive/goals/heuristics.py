"""Modular heuristics for translating goals into actions."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional, Pattern

from .dag_store import GoalNode


ActionDeque = Deque[Dict[str, object]]
HeuristicFn = Callable[[GoalNode, re.Match[str]], ActionDeque]


@dataclass
class RegexHeuristic:
    """Wraps a compiled regex and an action builder."""

    name: str
    pattern: Pattern[str]
    builder: HeuristicFn

    def try_apply(self, goal: GoalNode) -> Optional[ActionDeque]:
        description = (goal.description or "").strip()
        if not description:
            return None
        match = self.pattern.search(description)
        if not match:
            return None
        return self.builder(goal, match)


class HeuristicRegistry:
    """Registry of heuristics evaluated in order of registration."""

    def __init__(self) -> None:
        self._heuristics: list[RegexHeuristic] = []

    def register_regex(
        self,
        name: str,
        pattern: str,
        builder: HeuristicFn,
        *,
        flags: int | None = None,
    ) -> None:
        compiled = re.compile(pattern, flags or 0)
        self._heuristics.append(RegexHeuristic(name=name, pattern=compiled, builder=builder))

    def extend(self, heuristics: Iterable[RegexHeuristic]) -> None:
        self._heuristics.extend(heuristics)

    def match(self, goal: GoalNode) -> Optional[ActionDeque]:
        for heuristic in self._heuristics:
            try:
                actions = heuristic.try_apply(goal)
            except Exception:
                continue
            if actions:
                return actions
        return None


def default_heuristics() -> HeuristicRegistry:
    """Factory configuring the built-in heuristics."""

    registry = HeuristicRegistry()

    # Heuristic for learning concepts with more robust pattern matching.
    concept_pattern = (
        r"(?is)\bappr(?:e|é)ndre\s+(?:le|la|l')\s+concept\s*(?:de|d[eé])?\s*[«\"“”']?\s*"
        r"([^«»\"“”']+)\s*[»\"“”']?(?:\.|$)"
    )

    def build_concept(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        concept = match.group(1).strip()
        priority = getattr(goal, "priority", 0.6)
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "learn_concept",
                "payload": {"concept": concept, "goal_id": goal.id},
                "priority": min(1.0, priority + 0.2),
            }
        )
        return actions

    registry.register_regex("learn_concept", concept_pattern, build_concept)

    # Broader contradiction resolution heuristic with flexible separators/case.
    contradiction_pattern = (
        r"(?is)résoudre\s+contradiction\s*[«\"“”']?\s*([^,;«»\"“”']+)\s*[;,]\s*"
        r"([^»«\"“”']+)\s*[»\"“”']?"
    )

    def build_contradiction(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        subject = match.group(1).strip()
        relation = match.group(2).strip()
        base_priority = getattr(goal, "priority", 0.6)
        actions: ActionDeque = deque()
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
                    "subject": subject,
                    "relation": relation,
                    "value": "(à confirmer)",
                    "confidence": 0.6,
                },
                "priority": base_priority,
            }
        )
        return actions

    registry.register_regex("resolve_contradiction", contradiction_pattern, build_contradiction)

    # Recognise declarative "X est un concept" phrasing.
    declarative_pattern = r"(?is)\b(.+?)\s+est\s+(?:un|une|le|la|l')\s+concept\b"

    def build_declarative(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        concept = match.group(1).strip(" «»\"“”'")
        priority = getattr(goal, "priority", 0.5)
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "learn_concept",
                "payload": {"concept": concept, "goal_id": goal.id, "mode": "declarative"},
                "priority": min(1.0, priority + 0.15),
            }
        )
        return actions

    registry.register_regex("declarative_concept", declarative_pattern, build_declarative)

    return registry
