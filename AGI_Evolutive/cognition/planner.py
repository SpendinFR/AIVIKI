import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from AGI_Evolutive.retrieval.rag5.pipeline import RAGPipeline

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
        arch = architecture or self.architecture
        if arch is None:
            return out
        try:
            intent = _get_field(frame, "intent")
            if intent not in {"ask", "request", "summarize"}:
                return out
            rag_component = getattr(arch, "rag", None)
            if rag_component is None and isinstance(arch, RAGPipeline):
                rag_component = arch
            if rag_component is None:
                return out
            if isinstance(rag_component, RAGPipeline) or hasattr(rag_component, "ask"):
                rag_iface = rag_component
            else:
                return out
            text = _get_field(frame, "text")
            if not text:
                text = _get_field(frame, "surface_form")
            if not text:
                return out
            rag_out = rag_iface.ask(str(text))
            out["rag_out"] = rag_out
            if isinstance(rag_out, dict) and rag_out.get("status") == "ok":
                out["rag_signals"] = _rag_signals(rag_out)
                citations = rag_out.get("citations") or []
                grounded = []
                for cite in citations:
                    if not isinstance(cite, dict):
                        continue
                    grounded.append(
                        {
                            "doc_id": cite.get("doc_id"),
                            "offsets": (cite.get("start"), cite.get("end")),
                            "snippet": cite.get("snippet"),
                            "score": cite.get("score"),
                        }
                    )
                out["grounded_context"] = grounded
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
                    self._save()
                    return step
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
