import json
import pathlib
import sys
from types import SimpleNamespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.io.action_interface import Action, ActionInterface, CognitiveDomain


class _DummyMetacog:
    def __init__(self):
        self.calls = []

    def trigger_reflection(self, *, trigger, domain, urgency, depth):
        # record inputs to assert fallback behaviour
        self.calls.append({
            "trigger": trigger,
            "domain": domain,
            "urgency": urgency,
            "depth": depth,
        })
        return SimpleNamespace(duration=1.0, quality_score=0.75)


def test_reflect_action_falls_back_to_reasoning_domain(tmp_path):
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(tmp_path / "out"),
    )
    dummy = _DummyMetacog()
    # bind without exposing CognitiveDomain attribute to force fallback
    interface.bind(metacog=dummy)

    act = Action(id="a1", type="reflect", payload={}, priority=0.5)

    result = interface._h_reflect(act)

    assert result["ok"] is True
    assert dummy.calls, "Metacognition module should be invoked"
    call = dummy.calls[0]
    assert call["domain"] is CognitiveDomain.REASONING


def test_simulate_fallback_when_simulator_missing(tmp_path):
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(tmp_path / "out"),
    )

    act = Action(id="sim1", type="simulate", payload={"desc": "test"}, priority=0.5)

    result = interface._h_simulate(act)

    assert result["ok"] is True
    assert result["simulated"] is False
    assert result["reason"] == "simulator_unavailable"
    assert result["success"] is False


def test_execute_log_action_records_entry(tmp_path):
    out_dir = tmp_path / "out"
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(out_dir),
    )

    result = interface.execute({"type": "log", "text": "Bonjour", "level": "info"})

    assert result["ok"] is True

    log_path = out_dir / "log_entries.jsonl"
    assert log_path.exists(), "log handler should persist entries"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "log file should contain at least one entry"
    payload = json.loads(lines[-1])
    assert payload["text"] == "Bonjour"


def test_plan_step_handler_creates_record(tmp_path):
    out_dir = tmp_path / "out"
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(out_dir),
    )

    act = Action(
        id="ps1",
        type="plan_step",
        payload={"description": "Lister les ressources", "goal_id": "g1"},
        priority=0.5,
    )

    result = interface._h_plan_step(act)

    assert result["ok"] is True
    assert result["description"] == "Lister les ressources"

    record_path = out_dir / "plan_steps.jsonl"
    assert record_path.exists()
    entries = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["description"] == "Lister les ressources"
