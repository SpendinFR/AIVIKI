import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional

class Planner:
    """Light-weight goal/plan manager persisting to disk."""

    def __init__(self, path: str = "data/planner.json"):
        self.path = path
        self.state: Dict[str, Any] = {"plans": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                self.state = {"plans": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

    def plan_for_goal(self, goal_id: str, description: str) -> Dict[str, Any]:
        plan = self.state.setdefault("plans", {}).get(goal_id)
        if not plan:
            plan = {
                "id": goal_id,
                "description": description,
                "created_at": time.time(),
                "steps": []
            }
            self.state["plans"][goal_id] = plan
            self._save()
        else:
            if description and plan.get("description") != description:
                plan["description"] = description
                self._save()
        return plan

    def add_step(self, goal_id: str, desc: str) -> Dict[str, Any]:
        current_desc = self.state.get("plans", {}).get(goal_id, {}).get("description", desc)
        plan = self.plan_for_goal(goal_id, current_desc)
        step = {
            "id": f"{goal_id}_{uuid.uuid4().hex[:6]}",
            "desc": desc,
            "status": "pending",
            "created_at": time.time(),
            "last_update": time.time(),
            "history": []
        }
        plan["steps"].append(step)
        self._save()
        return step

    def pop_next_action(self, goal_id: str) -> Optional[Dict[str, Any]]:
        plan = self.state.get("plans", {}).get(goal_id)
        if not plan:
            return None
        for step in plan.get("steps", []):
            if step.get("status") == "pending":
                step["status"] = "in_progress"
                step["last_update"] = time.time()
                step.setdefault("history", []).append({"ts": time.time(), "event": "started"})
                self._save()
                return step
        return None

    def mark_action_done(self, goal_id: str, step_id: str, success: bool = True):
        plan = self.state.get("plans", {}).get(goal_id)
        if not plan:
            return
        for step in plan.get("steps", []):
            if step.get("id") == step_id:
                step["status"] = "done" if success else "blocked"
                step["last_update"] = time.time()
                step.setdefault("history", []).append({"ts": time.time(), "event": "completed", "success": success})
                break
        self._save()

    def pending_goals(self) -> List[Dict[str, Any]]:
        return [p for p in self.state.get("plans", {}).values() if any(s.get("status") == "pending" for s in p.get("steps", []))]
