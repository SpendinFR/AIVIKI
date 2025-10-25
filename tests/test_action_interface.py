import json
import pathlib
import sys
from types import SimpleNamespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AGI_Evolutive.io.action_interface import Action, ActionInterface, CognitiveDomain
from AGI_Evolutive.self_improver.skill_acquisition import SkillSandboxManager


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


class _MemoryStub:
    def __init__(self):
        self.records = []

    def search(self, query, top_k=8):
        return [
            {"kind": "knowledge", "content": f"Info sur {query}", "importance": 0.8},
            {"kind": "example", "content": f"Exemple pratique {query}"},
        ]

    def add_memory(self, payload):
        self.records.append(payload)
        return f"mem-{len(self.records)}"


class _LanguageStub:
    def reply(self, intent, data, pragmatic):
        topic = data.get("topic", "action")
        return f"Plan validé pour {topic}."


def test_unknown_action_goes_through_skill_sandbox(tmp_path):
    out_dir = tmp_path / "out"
    interface = ActionInterface(
        path_log=str(tmp_path / "actions.jsonl"),
        output_dir=str(out_dir),
    )

    memory = _MemoryStub()
    language = _LanguageStub()
    skills = SkillSandboxManager(
        storage_dir=str(tmp_path / "skills"),
        min_trials=2,
        success_threshold=0.4,
        run_async=False,
        approval_required=True,
        training_interval=0.0,
    )
    skills.bind(memory=memory, language=language)

    interface.bind(memory=memory, language=language, skills=skills)

    act = Action(
        id="skill-1",
        type="perform_magic",
        payload={"description": "Créer un tour de magie", "requirements": ["magie", "présentation"]},
        priority=0.5,
    )

    first = skills.handle_simulation(act, interface)

    assert first["ok"] is False
    assert first["reason"] in {"skill_waiting_user_approval", "skill_training_in_progress"}

    approve = interface.execute(
        {
            "type": "review_skill_candidate",
            "payload": {
                "action_type": "perform_magic",
                "decision": "approve",
                "reviewer": "tester",
            },
        }
    )

    assert approve["ok"] is True
    assert approve["skill"]["status"] == "active"

    second = interface.execute({"type": "perform_magic", "payload": {"audience": "demo"}})

    assert second["ok"] is True
    assert "summary" in second
