import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive import planning as planning_module
from AGI_Evolutive.planning import HTNPlanner, generate_structured_plan
from AGI_Evolutive.reasoning.structures import TaskNode


class _DummyBeliefs:
    def query(self, *args, **kwargs):
        return ["belief"]


@pytest.fixture
def planner() -> HTNPlanner:
    planner = HTNPlanner(_DummyBeliefs(), object())
    template = TaskNode(
        name="stabiliser_api",
        actions=["Collecter logs", "Redémarrer service"],
    )
    planner.register_template("stabiliser_api", template)
    return planner


def test_generate_structured_plan_uses_llm(monkeypatch, planner):
    captured = {}

    def _fake_call(spec_key: str, **kwargs):
        assert spec_key == "planning_overview"
        captured["payload"] = kwargs["input_payload"]
        return {
            "plan_name": "Stabiliser l'API",
            "steps": [
                {
                    "id": "analyse",
                    "description": "Collecter logs",
                    "depends_on": [],
                    "priority": 1,
                },
                {
                    "id": "mitigation",
                    "description": "Redémarrer service",
                    "depends_on": ["analyse"],
                    "priority": 2,
                },
            ],
            "risks": ["Logs incomplets"],
            "fallback_used": False,
            "confidence": 0.75,
            "notes": "On suit la procédure standard.",
        }

    monkeypatch.setattr(planning_module, "try_call_llm_dict", _fake_call)

    result = generate_structured_plan(
        "stabiliser_api",
        context={"severity": "haut"},
        planner=planner,
    )

    payload = captured["payload"]
    assert payload["goal"] == "stabiliser_api"
    assert payload["heuristic_plan"] == ["Collecter logs", "Redémarrer service"]
    assert "stabiliser_api" in payload["available_templates"]

    assert result["source"] == "llm"
    assert result["plan_name"] == "Stabiliser l'API"
    assert result["confidence"] == pytest.approx(0.75)
    assert len(result["steps"]) == 2
    assert result["steps"][1]["depends_on"] == ["analyse"]
    assert result["fallback_plan"] == ["Collecter logs", "Redémarrer service"]
    assert result["text_steps"][0].startswith("1.")


def test_generate_structured_plan_fallback(monkeypatch, planner):
    monkeypatch.setattr(planning_module, "try_call_llm_dict", lambda *_, **__: None)

    result = generate_structured_plan("stabiliser_api", planner=planner)

    assert result["source"] == "heuristic"
    assert result["fallback_used"] is True
    assert result["steps"][0]["description"] == "Collecter logs"
    assert result["text_steps"] == ["Collecter logs", "Redémarrer service"]
    assert result.get("fallback_reason") == "llm_unavailable"


def test_generate_structured_plan_llm_requests_fallback(monkeypatch, planner):
    def _fake_call(*_, **__):
        return {
            "plan_name": "stabiliser_api",
            "fallback_used": True,
            "confidence": 0.2,
            "notes": "Utiliser le plan heuristique.",
        }

    monkeypatch.setattr(planning_module, "try_call_llm_dict", _fake_call)

    result = generate_structured_plan("stabiliser_api", planner=planner)

    assert result["source"] == "heuristic"
    assert result["fallback_reason"] == "llm_requested_fallback"
    assert result["confidence"] == pytest.approx(0.2)
    assert result["notes"] == "Utiliser le plan heuristique."
