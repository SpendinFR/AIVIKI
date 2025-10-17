import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize

from AGI_Evolutive.core.config import cfg


def _rag_signals(rag_out: Dict[str, Any]) -> Dict[str, float]:
    if not rag_out or rag_out.get("status") != "ok":
        return {"rag_top1": 0.0, "rag_docs": 0.0, "rag_mean": 0.0, "rag_diversity": 0.0}
    cites = rag_out.get("citations") or []
    if not cites:
        return {"rag_top1": 0.0, "rag_docs": 0.0, "rag_mean": 0.0, "rag_diversity": 0.0}
    scores = [c.get("score", 0.0) for c in cites]
    top1 = max(scores) if scores else 0.0
    mean = sum(scores) / max(1, len(scores))
    div = len({c.get("doc_id") for c in cites if isinstance(c, dict)}) / max(1, len(cites))
    return {
        "rag_top1": float(top1),
        "rag_docs": float(len(cites)),
        "rag_mean": float(mean),
        "rag_diversity": float(div),
    }

_PLANS = cfg()["PLANS_PATH"]

DEFAULT_STOP_RULES: Dict[str, Any] = {"max_options": 3, "max_seconds": 900}


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ensure_dict_field(obj: Any, key: str) -> Dict[str, Any]:
    if isinstance(obj, dict):
        value = obj.get(key)
        if not isinstance(value, dict):
            value = {}
            obj[key] = value
        return value
    value = getattr(obj, key, None)
    if not isinstance(value, dict):
        value = {}
        setattr(obj, key, value)
    return value


def _set_field(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


class Planner:
    """Persisted goal planner managing micro-actions."""

    def __init__(self, architecture: Optional[Any] = None) -> None:
        self._lock = threading.RLock()
        self.state: Dict[str, Any] = {"plans": {}, "updated_at": time.time()}
        self.architecture: Optional[Any] = architecture
        self._load()

    def _maybe_preplan_with_rag(self, frame: Any, architecture: Optional[Any]) -> Dict[str, Any]:
        out = {"rag_out": None, "rag_signals": {}, "grounded_context": []}
        try:
            arch = architecture or self.architecture
            rag = getattr(arch, "rag", None)
            if getattr(frame, "intent", None) in {"ask", "request", "summarize"} and rag:
                text = _get_field(frame, "text") or _get_field(frame, "surface_form")
                if not text:
                    return out
                rag_out = rag.ask(str(text))
                out["rag_out"] = rag_out
                if isinstance(rag_out, dict) and rag_out.get("status") == "ok":
                    cites = rag_out.get("citations") or []
                    scores = [c.get("score", 0.0) for c in cites if isinstance(c, dict)]
                    top1 = max(scores) if scores else 0.0
                    mean = sum(scores) / max(1, len(scores)) if scores else 0.0
                    div = (
                        len({c.get("doc_id") for c in cites if isinstance(c, dict)})
                        / max(1, len(cites))
                        if cites
                        else 0.0
                    )
                    out["rag_signals"] = {
                        "rag_top1": float(top1),
                        "rag_docs": float(len(cites)),
                        "rag_mean": float(mean),
                        "rag_diversity": float(div),
                    }
                    out["grounded_context"] = [
                        {
                            "doc_id": c.get("doc_id"),
                            "offsets": (c.get("start"), c.get("end")),
                            "snippet": c.get("snippet"),
                            "score": c.get("score"),
                        }
                        for c in cites
                        if isinstance(c, dict)
                    ]
        except Exception:
            pass
        return out

    def _load(self) -> None:
        if os.path.exists(_PLANS):
            try:
                with open(_PLANS, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    self.state = data
            except Exception:
                pass

    def _save(self) -> None:
        with self._lock:
            self.state["updated_at"] = time.time()
            os.makedirs(os.path.dirname(_PLANS), exist_ok=True)
            with open(_PLANS, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(self.state), f, ensure_ascii=False, indent=2)

    def plan_for_goal(
        self, goal_id: str, description: str, steps: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        with self._lock:
            plan = self.state["plans"].get(goal_id)
            if not plan:
                plan = {
                    "goal_id": goal_id,
                    "description": description,
                    "steps": steps or [],
                    "created_at": time.time(),
                }
                self.state["plans"][goal_id] = plan
                self._save()
            return plan

    def _normalise_stop_rules(self, stop_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        rules = dict(DEFAULT_STOP_RULES)
        if isinstance(stop_rules, dict):
            for key, value in stop_rules.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    rules[key] = value
                else:
                    rules[key] = value
        return rules

    def frame(
        self,
        goal: Any,
        stop_rules: Optional[Dict[str, Any]] = None,
        architecture: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Construit un cadre de planification pour un goal donné.

        Le ``goal`` peut être un ``GoalNode``, un dict ou simplement un identifiant.
        ``stop_rules`` est merge avec :data:`DEFAULT_STOP_RULES`.
        """

        rules = self._normalise_stop_rules(stop_rules)
        goal_id: Optional[str] = None
        description: Optional[str] = None
        context: Dict[str, Any] = {}

        if isinstance(goal, dict):
            goal_id = goal.get("id") or goal.get("goal_id") or goal.get("name")
            description = goal.get("description") or goal.get("desc")
            context = dict(goal.get("context", {})) if isinstance(goal.get("context"), dict) else {}
        elif hasattr(goal, "id"):
            goal_id = getattr(goal, "id", None)
            description = getattr(goal, "description", None)
        elif isinstance(goal, str):
            goal_id = goal

        plan: Optional[Dict[str, Any]] = None
        with self._lock:
            if goal_id:
                plan = self.state["plans"].get(goal_id)
        if goal_id and plan is None:
            desc = description or (goal_id.replace("_", " ") if isinstance(goal_id, str) else "")
            plan = self.plan_for_goal(goal_id, desc)

        frame: Dict[str, Any] = {
            "goal_id": goal_id,
            "description": description or (plan.get("description") if isinstance(plan, dict) else None),
            "stop_rules": rules,
        }
        if context:
            frame["context"] = context
        if plan:
            frame["plan"] = plan

        if architecture is not None:
            try:
                rag_bundle = self._maybe_preplan_with_rag(goal, architecture)
                if isinstance(rag_bundle, dict):
                    signals = rag_bundle.get("rag_signals") or {}
                    if signals:
                        frame.setdefault("signals", {}).update(signals)
                    grounded = rag_bundle.get("grounded_context")
                    if grounded:
                        frame.setdefault("context", {})["grounded_evidence"] = grounded
            except Exception:
                pass

        return frame

    def _infer_action_type(self, desc: str) -> str:
        text = (desc or "").lower()
        if any(token in text for token in ("question", "ask", "demande", "poser")):
            return "message_user"
        if any(token in text for token in ("observer", "observe", "regarder")):
            return "simulate"
        if any(token in text for token in ("écrire", "note", "consigner", "journaliser")):
            return "write_memory"
        if any(token in text for token in ("plan", "structurer", "organiser")):
            return "plan"
        if any(token in text for token in ("analyser", "analyze", "analyser")):
            return "reflect"
        return "reflect"

    def _step_to_action(self, goal_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        desc = step.get("desc", "")
        action_spec = step.get("action") if isinstance(step.get("action"), dict) else {}
        payload = dict(action_spec.get("payload", {}))
        payload.setdefault("goal_id", goal_id)
        payload.setdefault("step_id", step.get("id"))
        if "context" in step and isinstance(step["context"], dict):
            payload.setdefault("context", step["context"])

        act_type = action_spec.get("type") or step.get("type") or self._infer_action_type(desc)
        action = {
            "type": act_type,
            "payload": payload,
        }
        if "mode" in action_spec:
            action["mode"] = action_spec["mode"]

        priority = step.get("priority")
        if not isinstance(priority, (int, float)):
            priority = payload.get("priority", 0.5)

        return {
            "id": step.get("id"),
            "goal_id": goal_id,
            "desc": desc,
            "status": step.get("status", "doing"),
            "priority": float(priority) if isinstance(priority, (int, float)) else 0.5,
            "action": action,
        }

    def plan(self, frame: Any, architecture: Optional[Any] = None) -> Dict[str, Any]:
        if architecture is not None:
            self.architecture = architecture
        arch = architecture or self.architecture
        rag_bundle = self._maybe_preplan_with_rag(frame, arch)

        if rag_bundle["rag_out"] and rag_bundle["rag_out"].get("status") == "refused":
            plan = {
                "type": "clarify",
                "reason": rag_bundle["rag_out"].get("reason"),
                "questions": [
                    "Peux-tu préciser la période/la source attendue ?",
                    "As-tu un lien ou document de référence ?",
                ],
            }
            _set_field(frame, "plan", plan)
            return plan

        if rag_bundle["grounded_context"]:
            context = _ensure_dict_field(frame, "context")
            context["grounded_evidence"] = rag_bundle["grounded_context"]
            signals = _ensure_dict_field(frame, "signals")
            signals.update(rag_bundle["rag_signals"])

        plan = _get_field(frame, "plan")
        if isinstance(plan, dict):
            return plan
        return {}

    def add_step(self, goal_id: str, desc: str) -> str:
        with self._lock:
            plan = self.state["plans"].setdefault(
                goal_id, {"goal_id": goal_id, "description": "", "steps": []}
            )
            step_id = f"s{len(plan['steps']) + 1}"
            plan["steps"].append({"id": step_id, "desc": desc, "status": "todo"})
            self._save()
            return step_id

    def pop_next_action(self, goal_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            plan = self.state["plans"].get(goal_id)
            if not plan:
                return None
            for step in plan["steps"]:
                if step["status"] == "todo":
                    step["status"] = "doing"
                    step["last_emitted_at"] = time.time()
                    action = self._step_to_action(goal_id, step)
                    self._save()
                    return action
            return None

    def mark_action_done(self, goal_id: str, step_id: str, success: bool = True):
        with self._lock:
            plan = self.state.get("plans", {}).get(goal_id)
            if not plan:
                return
            for step in plan.get("steps", []):
                if step.get("id") == step_id:
                    step["status"] = "done" if success else "blocked"
                    step["last_update"] = time.time()
                    step.setdefault("history", []).append(
                        {"ts": time.time(), "event": "completed", "success": success}
                    )
                    break
            self._save()

    def pending_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            pending_statuses = {"todo", "doing", "blocked"}
            return [
                p
                for p in self.state.get("plans", {}).values()
                if any(s.get("status") in pending_statuses for s in p.get("steps", []))
            ]
    def simulate_action(self, step_desc: str) -> Dict[str, Any]:
        length = len(step_desc)
        success_prob = max(0.3, min(0.9, 1.0 - (length / 200.0)))
        expected_time = min(10.0, 1.0 + length / 80.0)
        return {"success_prob": success_prob, "expected_time": expected_time}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def plans_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {gid: dict(plan) for gid, plan in self.state["plans"].items()}
