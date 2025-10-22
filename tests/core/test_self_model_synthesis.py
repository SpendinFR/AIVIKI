import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AGI_Evolutive.core import config as config_module
from AGI_Evolutive.core.self_model import SelfModel


def _prepare_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    data = {
        "DATA_DIR": str(tmp_path / "data"),
        "MEM_DIR": str(tmp_path / "mem"),
        "SELF_PATH": str(tmp_path / "self.json"),
        "SELF_VERSIONS_DIR": str(tmp_path / "versions"),
        "VECTOR_DIR": str(tmp_path / "vec"),
        "LOGS_DIR": str(tmp_path / "logs"),
    }
    cfg_path.write_text(json.dumps(data))
    config_module.load_config.cache_clear()
    config_module._cfg = None
    config_module.load_config(str(cfg_path))


def test_self_model_build_synthesis(tmp_path):
    _prepare_config(tmp_path)
    model = SelfModel()
    model.ensure_identity_paths()

    model.set_identity_patch(
        {
            "preferences": {"likes": ["music", "learning"], "values": ["curiosity", "kindness"]},
            "ideals": ["growth"],
            "principles": [{"key": "care", "desc": "Be caring"}],
            "purpose": {
                "mission": "Support users",
                "ultimate": "Empower",  # override default
                "current_goal": {"id": "g1", "description": "Assist"},
                "daily_focus": ["support"],
            },
            "agenda": {
                "today": ["chat"],
                "tomorrow": ["review"],
                "horizon": ["learn"],
            },
            "achievements": {"recent": [{"summary": "Completed task"}], "milestones": ["Shipped feature"]},
            "narrative": {"recent_memories": ["Chat with user"], "summaries": ["Weekly digest"]},
            "social": {"interactions": [{"user": "Alice", "topic": "memory"}]},
            "reflections": {"past": ["Improved"], "present": {"summary": "Confident", "confidence": 0.7}},
            "beliefs": {"index": {"memory": {"stance": "important", "confidence": 0.9}}},
        }
    )
    model.update_state(
        emotions={"valence": 0.6, "arousal": 0.4},
        cognition={"thinking": 0.5, "load": 0.3},
        doubts=[{"topic": "scalability"}],
    )
    model.attach_selfhood(traits={"self_trust": 0.8}, phase="builder", claims={})

    summary = model.build_synthesis(max_items=2)

    assert summary["identity"]["name"]
    assert summary["likes"] == ["music", "learning"][:2]
    assert summary["values"][0] == "curiosity"
    assert summary["principles"][0]["key"] == "care"
    assert summary["ideals"] == ["growth"]
    assert summary["beliefs"][0]["topic"] == "memory"
    assert summary["ultimate_goal"] == "Empower"
    assert summary["agenda"]["today"] == ["chat"]
    assert summary["achievements"]["recent"][0]["summary"] == "Completed task"
    assert summary["memories"]["recent"] == ["Chat with user"]
    assert summary["social"]["interactions"][0]["user"] == "Alice"
    assert summary["self_judgment"]["history"] == ["Improved"]
    assert summary["state"]["emotions"]["valence"] == 0.6
