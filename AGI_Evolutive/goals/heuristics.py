"""Modular heuristics for translating goals into actions."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional, Pattern

from .dag_store import GoalNode


ActionDeque = Deque[Dict[str, object]]
HeuristicFn = Callable[[GoalNode, re.Match[str]], ActionDeque]


def _topic_from_match(goal: GoalNode, match: re.Match[str]) -> str:
    raw = ""
    try:
        raw = match.group(1)
    except IndexError:
        raw = ""
    topic = (raw or "").strip()
    if topic:
        return topic
    description = goal.description or ""
    inner = re.search(r"«\s*([^»]+)\s*»", description)
    if inner:
        return inner.group(1).strip()
    return description.strip()


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

    diagnose_pattern = r"(?is)diagnostiquer\s+ma\s+compréhension.*«([^»]+)»"

    def build_diagnose(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.5) or 0.5
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "reflect",
                "payload": {
                    "goal_id": goal.id,
                    "hint": f"Analyser les métriques métacognitives liées à « {topic} ».",
                },
                "priority": max(0.1, priority),
            }
        )
        actions.append(
            {
                "type": "search_memory",
                "payload": {
                    "goal_id": goal.id,
                    "query": topic,
                    "hint": "Retrouver observations et traces existantes",
                },
                "priority": max(0.1, priority * 0.95),
            }
        )
        return actions

    registry.register_regex("diagnose_comprehension", diagnose_pattern, build_diagnose)

    solicit_pattern = r"(?is)solliciter\s+(?:le\s+)?questionmanager.*«([^»]+)»"

    def build_solicit(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.55) or 0.55
        question = (
            f"Quelles informations me manquent pour progresser sur « {topic} » ?"
        )
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "ask",
                "payload": {
                    "goal_id": goal.id,
                    "question": question,
                    "source": "goal_system",
                    "type": "clarifying",
                },
                "priority": min(1.0, priority + 0.1),
            }
        )
        return actions

    registry.register_regex("solicit_question_manager", solicit_pattern, build_solicit)

    explore_inbox_pattern = r"(?is)explorer\s+l'inbox.*«([^»]+)»"

    def build_explore_inbox(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.5) or 0.5
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "scan_inbox",
                "payload": {"goal_id": goal.id, "topic": topic, "limit": 5},
                "priority": min(1.0, priority + 0.05),
            }
        )
        actions.append(
            {
                "type": "reflect",
                "payload": {
                    "goal_id": goal.id,
                    "hint": f"Synthétiser les signaux de l'inbox liés à « {topic} ».",
                },
                "priority": priority,
            }
        )
        return actions

    registry.register_regex("explore_inbox_structural", explore_inbox_pattern, build_explore_inbox)

    consign_pattern = r"(?is)consigner\s+l'hypothèse.*«([^»]+)»"

    def build_consign(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.58) or 0.58
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "write_memory",
                "payload": {
                    "kind": "hypothesis",
                    "content": f"Hypothèse de travail consolidée pour « {topic} ».",
                    "meta": {"goal_id": goal.id},
                },
                "priority": min(1.0, priority + 0.05),
            }
        )
        return actions

    registry.register_regex("consign_hypothesis", consign_pattern, build_consign)

    identify_pattern = r"(?is)identifier\s+deux\s+questions.*«([^»]+)»"

    def build_identify_questions(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.52) or 0.52
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "reflect",
                "payload": {
                    "goal_id": goal.id,
                    "hint": f"Formuler deux questions fortes pour « {topic} ».",
                },
                "priority": max(0.1, priority),
            }
        )
        return actions

    registry.register_regex("identify_priority_questions", identify_pattern, build_identify_questions)

    submit_pattern = r"(?is)soumettre\s+les\s+questions.*questionmanager.*«([^»]+)»"

    def build_submit_questions(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.55) or 0.55
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "ask",
                "payload": {
                    "goal_id": goal.id,
                    "question": f"Soumission des questions préparées sur « {topic} ».",
                    "source": "goal_system",
                    "type": "follow_up",
                },
                "priority": min(1.0, priority + 0.08),
            }
        )
        return actions

    registry.register_regex("submit_questions", submit_pattern, build_submit_questions)

    capitalise_pattern = r"(?is)capitaliser\s+les\s+réponses.*questionmanager.*«([^»]+)»"

    def build_capitalise(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.57) or 0.57
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "write_memory",
                "payload": {
                    "kind": "question_manager_summary",
                    "content": f"Synthèse des réponses du QuestionManager pour « {topic} ».",
                    "meta": {"goal_id": goal.id},
                },
                "priority": min(1.0, priority + 0.05),
            }
        )
        return actions

    registry.register_regex("capitalise_qm_responses", capitalise_pattern, build_capitalise)

    collect_inbox_pattern = r"(?is)récolter\s+trois\s+éléments.*inbox.*«([^»]+)»"

    def build_collect_inbox(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.5) or 0.5
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "scan_inbox",
                "payload": {"goal_id": goal.id, "topic": topic, "limit": 3},
                "priority": min(1.0, priority + 0.06),
            }
        )
        return actions

    registry.register_regex("collect_inbox_signals", collect_inbox_pattern, build_collect_inbox)

    synth_inbox_pattern = r"(?is)synthétiser\s+les\s+apports.*inbox.*«([^»]+)»"

    def build_synth_inbox(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.55) or 0.55
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "write_memory",
                "payload": {
                    "kind": "inbox_synthesis",
                    "content": f"Résumé des apports inbox pour « {topic} ».",
                    "meta": {"goal_id": goal.id},
                },
                "priority": min(1.0, priority + 0.05),
            }
        )
        return actions

    registry.register_regex("synthesise_inbox", synth_inbox_pattern, build_synth_inbox)

    update_progress_pattern = r"(?is)mettre\s+à\s+jour\s+la\s+progression\s+du\s+but.*«([^»]+)»"

    def build_update_progress(goal: GoalNode, match: re.Match[str]) -> ActionDeque:
        topic = _topic_from_match(goal, match)
        priority = getattr(goal, "priority", 0.52) or 0.52
        actions: ActionDeque = deque()
        actions.append(
            {
                "type": "reflect",
                "payload": {
                    "goal_id": goal.id,
                    "hint": f"Aligner la progression du but parent « {topic} » avec les nouvelles données.",
                },
                "priority": max(0.1, priority),
            }
        )
        return actions

    registry.register_regex("update_parent_progress", update_progress_pattern, build_update_progress)

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
