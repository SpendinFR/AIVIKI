from collections import deque

import pytest

from AGI_Evolutive.goals import GoalSystem


@pytest.fixture()
def goal_system(tmp_path):
    data_path = tmp_path / "goal_intentions.json"
    persist_path = tmp_path / "goals.json"
    dashboard_path = tmp_path / "dashboard.json"
    system = GoalSystem(
        persist_path=str(persist_path),
        dashboard_path=str(dashboard_path),
        intention_data_path=str(data_path),
    )
    yield system
    # cleanup persisted files to avoid interference
    for path in (data_path, persist_path, dashboard_path):
        if path.exists():
            path.unlink()


def _deque_types(actions: deque) -> list[str]:
    return [action["type"] for action in actions]


def test_concept_heuristic_detects_quotes(goal_system):
    goal = goal_system.add_goal("Apprendre le concept « auto-éco-régulation »")
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert types[0] == "learn_concept"
    assert actions[0]["payload"]["concept"].lower() == "auto-éco-régulation"


def test_classifier_informs_actions(goal_system, tmp_path):
    goal = goal_system.add_goal("Planifier une étude sur la collaboration ouverte")
    # simulate execution success and feed back outcome
    goal_system.record_goal_outcome(
        goal.id, succeeded=True, executed_actions=[{"type": "plan"}]
    )
    # new goal with similar wording should now map to plan directly
    successor = goal_system.add_goal("Planifier une étude sur la collaboration ouverte")
    actions = goal_system._goal_to_actions(successor)
    types = _deque_types(actions)
    assert types[0] == "plan"


def test_probe_generated_on_low_confidence(goal_system):
    goal = goal_system.add_goal("Explorer un thème émergent")
    # feed conflicting labels to keep confidence low
    goal_system.intention_model.classifier.update("plan", goal.description)
    goal_system.intention_model.classifier.update("reflect", goal.description)
    goal_system.intent_confidence_threshold = 0.9
    actions = goal_system._goal_to_actions(goal)
    types = _deque_types(actions)
    assert types[0] == "probe_goal"
    assert types[1] == "reflect"
