from __future__ import annotations

import json
import os
from typing import Dict


class GoalDAG:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._state = {
            "id": "maintain_loop",
            "evi": 0.5,
            "progress": 0.1,
        }
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._state.update(data)
        except Exception:
            pass

    def choose_next_goal(self) -> Dict[str, float]:
        return dict(self._state)

    def update_goal(self, goal_id: str, evi: float, progress: float) -> None:
        self._state.update({"id": goal_id, "evi": float(evi), "progress": float(progress)})
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._state, f)
        except Exception:
            pass
