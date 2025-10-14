import os
import json
import time
from typing import Dict, Any

class SelfModel:
    """Keeps track of self-related attributes and applies proposals."""

    def __init__(self, path: str = "data/self_model.json"):
        self.path = path
        self.state: Dict[str, Any] = {
            "traits": {"adaptability": 0.6, "stability": 0.7},
            "history": []
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

    def apply_proposal(self, proposal: Dict[str, Any], policy_engine):
        record = {
            "ts": time.time(),
            "proposal": proposal
        }
        self.state.setdefault("history", []).append(record)
        self.state["history"] = self.state["history"][-200:]

        p_type = proposal.get("type")
        if p_type == "adjust_drive":
            drive = proposal.get("drive")
            target = proposal.get("target")
            if drive and target is not None:
                policy_engine.adjust_drive_target(drive, target)
        elif p_type == "planning_hint":
            policy_engine.register_hint(proposal)

        self._save()
