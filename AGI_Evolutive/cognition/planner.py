import json
import os
import time
from typing import Any, Dict, List, Optional

from core.config import cfg

_PLANS = cfg()["PLANS_PATH"]


class Planner:
    """Persisted goal planner managing micro-actions."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {"plans": {}, "updated_at": time.time()}
        self._load()

    def _load(self) -> None:
        if os.path.exists(_PLANS):
            try:
                with open(_PLANS, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self) -> None:
        self.state["updated_at"] = time.time()
        os.makedirs(os.path.dirname(_PLANS), exist_ok=True)
        with open(_PLANS, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def plan_for_goal(
        self, goal_id: str, description: str, steps: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if goal_id not in self.state["plans"]:
            self.state["plans"][goal_id] = {
                "goal_id": goal_id,
                "description": description,
                "steps": steps or [],
                "created_at": time.time(),
            }
            self._save()
        return self.state["plans"][goal_id]

    def add_step(self, goal_id: str, desc: str) -> str:
        plan = self.state["plans"].setdefault(goal_id, {"goal_id": goal_id, "description": "", "steps": []})
        step_id = f"s{len(plan['steps']) + 1}"
        plan["steps"].append({"id": step_id, "desc": desc, "status": "todo"})
        self._save()
        return step_id

    def pop_next_action(self, goal_id: str) -> Optional[Dict[str, Any]]:
        plan = self.state["plans"].get(goal_id)
        if not plan:
            return None
        for step in plan["steps"]:
            if step["status"] == "todo":
                step["status"] = "doing"
                self._save()
                return step
        return None

    def mark_action_done(self, goal_id: str, step_id: str, success: bool = True) -> None:
        plan = self.state["plans"].get(goal_id)
        if not plan:
            return
        for step in plan["steps"]:
            if step["id"] == step_id:
                step["status"] = "done" if success else "blocked"
                self._save()
                return

    def simulate_action(self, step_desc: str) -> Dict[str, Any]:
        length = len(step_desc)
        success_prob = max(0.3, min(0.9, 1.0 - (length / 200.0)))
        expected_time = min(10.0, 1.0 + length / 80.0)
        return {"success_prob": success_prob, "expected_time": expected_time}
