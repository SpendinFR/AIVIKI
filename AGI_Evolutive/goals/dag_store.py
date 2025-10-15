"""Persistent storage for goal DAG structures."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize


def _now() -> float:
    return time.time()


@dataclass
class GoalNode:
    id: str
    description: str
    criteria: List[str] = field(default_factory=list)
    progress: float = 0.0
    value: float = 0.5
    competence: float = 0.5
    curiosity: float = 0.2
    urgency: float = 0.3
    priority: float = 0.0
    status: str = "pending"
    created_by: str = "system"
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DagStore:
    """Minimal persistent DAG store for long-lived goals."""

    def __init__(self, persist_path: str, dashboard_path: str):
        self.persist_path = persist_path
        self.dashboard_path = dashboard_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.nodes: Dict[str, GoalNode] = {}
        self.active_goal_id: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    def add_goal(
        self,
        description: str,
        criteria: Optional[List[str]] = None,
        created_by: str = "system",
        value: float = 0.5,
        competence: float = 0.5,
        curiosity: float = 0.2,
        urgency: float = 0.3,
        parent_ids: Optional[List[str]] = None,
    ) -> GoalNode:
        gid = str(uuid.uuid4())[:8]
        node = GoalNode(
            id=gid,
            description=description,
            criteria=list(criteria or []),
            value=float(max(0.0, min(1.0, value))),
            competence=float(max(0.0, min(1.0, competence))),
            curiosity=float(max(0.0, min(1.0, curiosity))),
            urgency=float(max(0.0, min(1.0, urgency))),
            created_by=created_by,
            parent_ids=list(parent_ids or []),
        )
        self.nodes[gid] = node
        for pid in node.parent_ids:
            parent = self.nodes.get(pid)
            if parent and gid not in parent.child_ids:
                parent.child_ids.append(gid)
                parent.updated_at = _now()
        self._recompute_priority(node)
        self._persist()
        return node

    def link(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if not parent or not child:
            return
        if child_id not in parent.child_ids:
            parent.child_ids.append(child_id)
        if parent_id not in child.parent_ids:
            child.parent_ids.append(parent_id)
        parent.updated_at = _now()
        child.updated_at = _now()
        self._persist()

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[GoalNode]:
        node = self.nodes.get(goal_id)
        if not node:
            return None
        for key, value in updates.items():
            if not hasattr(node, key):
                continue
            setattr(node, key, value)
        node.updated_at = _now()
        self._recompute_priority(node)
        self._persist()
        return node

    def get_goal(self, goal_id: str) -> Optional[GoalNode]:
        return self.nodes.get(goal_id)

    def set_active(self, goal_id: Optional[str]) -> Optional[GoalNode]:
        if goal_id is None:
            self.active_goal_id = None
            self._export_dashboard()
            return None
        node = self.nodes.get(goal_id)
        if not node:
            return None
        self.active_goal_id = goal_id
        if node.status == "pending":
            node.status = "active"
        node.updated_at = _now()
        self._recompute_priority(node)
        self._persist()
        return node

    def get_active(self) -> Optional[GoalNode]:
        if self.active_goal_id:
            return self.nodes.get(self.active_goal_id)
        return None

    def complete_goal(self, goal_id: str, success: bool = True, note: str = "") -> None:
        node = self.nodes.get(goal_id)
        if not node:
            return
        node.status = "done" if success else "abandoned"
        node.progress = 1.0 if success else node.progress
        node.updated_at = _now()
        node.evidence.append({"t": _now(), "note": note, "success": success})
        if self.active_goal_id == goal_id:
            self.active_goal_id = None
        self._persist()

    # ------------------------------------------------------------------
    # Queries
    def topk(self, k: int = 5, only_pending: bool = True) -> List[GoalNode]:
        pool = [
            node
            for node in self.nodes.values()
            if (not only_pending) or node.status in {"pending", "active"}
        ]
        pool.sort(key=lambda n: n.priority, reverse=True)
        return pool[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    def _recompute_priority(self, node: GoalNode) -> None:
        value = max(0.0, min(1.0, node.value))
        urgency = max(0.0, min(1.0, node.urgency))
        curiosity = max(0.0, min(1.0, node.curiosity))
        competence = max(0.0, min(1.0, node.competence))
        base = 0.4 * value + 0.3 * urgency + 0.2 * curiosity + 0.1 * (1.0 - abs(0.5 - competence) * 2.0)
        if node.status not in {"pending", "active"}:
            base *= 0.2
        node.priority = float(max(0.0, min(1.0, base)))

    def _persist(self) -> None:
        payload = {
            "active_goal_id": self.active_goal_id,
            "nodes": {gid: node.to_dict() for gid, node in self.nodes.items()},
        }
        try:
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(payload), fh, ensure_ascii=False, indent=2)
        finally:
            self._export_dashboard()

    def _load(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return
        nodes = {}
        for gid, node_data in data.get("nodes", {}).items():
            try:
                nodes[gid] = GoalNode(**node_data)
            except TypeError:
                continue
        self.nodes = nodes
        self.active_goal_id = data.get("active_goal_id")

    def _export_dashboard(self) -> None:
        snapshot = {
            "t": _now(),
            "active_goal": self.get_active().to_dict() if self.get_active() else None,
            "top5": [node.to_dict() for node in self.topk(5, only_pending=False)],
            "counts": {
                "total": len(self.nodes),
                "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
                "active": sum(1 for n in self.nodes.values() if n.status == "active"),
                "done": sum(1 for n in self.nodes.values() if n.status == "done"),
            },
        }
        try:
            with open(self.dashboard_path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(snapshot), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass


class GoalDAG:
    """Lightweight view for quick goal sampling."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._state = {"id": "maintain_loop", "evi": 0.5, "progress": 0.1}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return
        if isinstance(data, dict):
            self._state.update(data)

    def choose_next_goal(self) -> Dict[str, Any]:
        return dict(self._state)

    def update_goal(self, goal_id: str, evi: float, progress: float) -> None:
        self._state.update({"id": goal_id, "evi": float(evi), "progress": float(progress)})
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(json_sanitize(self._state), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def bump_progress(self, delta: float = 0.01) -> float:
        cur = float(self._state.get("progress", 0.0))
        cur = min(1.0, max(0.0, cur + float(delta)))
        self._state["progress"] = cur
        try:
            self._save()  # si tu as déjà _save(); sinon ignore
        except Exception:
            try:
                with open(self.path, "w", encoding="utf-8") as fh:
                    json.dump(json_sanitize(self._state), fh, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return cur
