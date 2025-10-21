"""Tests for contextual question generation in the QuestionManager."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.core.question_manager import QuestionManager


class _DummyGoals:
    def __init__(self, description: Optional[str]) -> None:
        self._description = description

    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        if not self._description:
            return None
        return {"description": self._description}


class _DummyArch:
    def __init__(self, goal_description: Optional[str]) -> None:
        self.goals = _DummyGoals(goal_description)


def _pop_question_texts(qm: QuestionManager) -> list[str]:
    qm.maybe_generate_questions()
    pending = qm.pop_questions()
    return [item["text"] for item in pending]


def test_question_manager_anchors_on_active_goal_topic():
    arch = _DummyArch("Comprendre le concept « empathie » pour mieux aider l'utilisateur")
    qm = QuestionManager(arch)
    qm.record_information_need("goal_focus", 0.72)

    questions = _pop_question_texts(qm)

    assert questions, "A contextual question should be generated"
    assert any("empathie" in q.lower() for q in questions)


def test_question_manager_uses_metadata_topic_override():
    arch = _DummyArch("Explorer les émotions complexes")
    qm = QuestionManager(arch)
    qm.record_information_need(
        "evidence",
        0.65,
        metadata={"topic": "auto-régulation émotionnelle"},
    )

    questions = _pop_question_texts(qm)

    assert questions, "A contextual evidence question should be generated"
    assert any("auto-régulation" in q.lower() for q in questions)


def test_question_manager_falls_back_to_library_when_no_focus():
    arch = _DummyArch(None)
    qm = QuestionManager(arch)
    qm.record_information_need("success_metric", 0.5)

    questions = _pop_question_texts(qm)

    assert questions, "Library question should be available"
    assert any("indicateur" in q.lower() for q in questions)
