from types import SimpleNamespace

from AGI_Evolutive.self_improver.skill_acquisition import (
    SkillRequest,
    SkillSandboxManager,
    SkillTrial,
)


def test_list_skills_filters_and_formats(tmp_path):
    manager = SkillSandboxManager(storage_dir=str(tmp_path / "skills"), run_async=False)

    req_pending = SkillRequest(
        identifier="req-1",
        action_type="alpha_test",
        description="Alpha",
        payload={},
        created_at=1.0,
        status="awaiting_approval",
    )
    req_pending.trials = [
        SkillTrial(index=0, coverage=0.8, success=True),
        SkillTrial(index=1, coverage=0.6, success=False),
    ]
    req_pending.attempts = 2
    req_pending.successes = 1
    req_pending.requirements = ["alpha", "test"]

    req_failed = SkillRequest(
        identifier="req-2",
        action_type="beta_test",
        description="Beta",
        payload={},
        created_at=2.0,
        status="failed",
    )
    req_failed.trials = [
        SkillTrial(index=0, coverage=0.2, success=False),
    ]
    req_failed.attempts = 1
    req_failed.successes = 0
    req_failed.requirements = ["beta"]

    with manager._lock:  # type: ignore[attr-defined]
        manager._requests = {
            req_pending.action_type: req_pending,
            req_failed.action_type: req_failed,
        }

    awaiting = manager.list_skills(status="awaiting_approval", include_trials=True)
    assert len(awaiting) == 1
    entry = awaiting[0]
    assert entry["action_type"] == "alpha_test"
    assert entry["success_rate"] == 0.5
    assert len(entry["trials"]) == 2

    failed = manager.list_skills(status="failed")
    assert len(failed) == 1
    assert failed[0]["trial_count"] == 1


class _SimulatorStub:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        return {
            "success": True,
            "summary": "Essai validé dans le simulateur",
            "feedback": "Validation complète",
            "evidence": ["magie", "présentation"],
        }


def test_training_records_practice_feedback(tmp_path):
    manager = SkillSandboxManager(
        storage_dir=str(tmp_path / "skills"),
        run_async=False,
        min_trials=1,
        success_threshold=0.4,
        training_interval=0.0,
    )
    simulator = _SimulatorStub()
    manager.bind(simulator=simulator)

    act = SimpleNamespace(
        type="perform_magic",
        payload={"description": "Faire un tour de magie", "requirements": ["magie", "présentation"]},
        context={},
    )

    result = manager.handle_simulation(act, interface=None)

    assert result["ok"] is False
    assert result["reason"] in {"skill_waiting_user_approval", "skill_training_in_progress"}
    assert simulator.calls, "Simulator should be invoked during training"

    status = manager.status("perform_magic")
    assert status["status"] in {"awaiting_approval", "active"}
    assert status["success_rate"] >= 1.0

    trials = status.get("trials", [])
    assert trials, "Trials should be recorded with simulator feedback"
    trial = trials[0]
    assert trial["mode"] == "simulator"
    assert trial["success"] is True
    assert "Essai validé" in (trial.get("summary") or "")
    assert any("magie" in evidence.lower() for evidence in trial.get("evidence", []))
