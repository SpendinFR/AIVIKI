import os
import json
import time
from typing import Dict, Any

class PolicyEngine:
    """Stores lightweight policy directives and strategy hints."""

    def __init__(self, path: str = "data/policy.json"):
        self.path = path
        self.state: Dict[str, Any] = {
            "drive_targets": {},
            "hints": []
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

    def adjust_drive_target(self, drive: str, target: float):
        self.state.setdefault("drive_targets", {})[drive] = float(max(0.0, min(1.0, target)))
        self.state.setdefault("history", []).append({
            "ts": time.time(),
            "event": "drive_target",
            "drive": drive,
            "target": target
        })
        self.state["history"] = self.state["history"][-200:]
        self._save()

    def register_hint(self, hint: Dict[str, Any]):
        entry = dict(hint)
        entry["ts"] = time.time()
        self.state.setdefault("hints", []).append(entry)
        self.state["hints"] = self.state["hints"][-100:]
        self._save()
