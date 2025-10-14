from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import time
import json
import os
import uuid
import math


@dataclass
class GoalNode:
    id: str
    description: str
    criteria: List[str] = field(default_factory=list)
    progress: float = 0.0            # [0,1]
    value: float = 0.5               # [0,1] (utilité/valeur perçue)
    competence: float = 0.4          # [0,1] (estimation de capacité de réussite)
    curiosity: float = 0.2           # [0,1] (gain d'info anticipé)
    urgency: float = 0.3             # [0,1] (délais/risques)
    priority: float = 0.0            # [0,1] (score calculé)
    status: str = "pending"          # "pending"|"active"|"blocked"|"done"|"abandoned"
    created_by: str = "system"       # "system"|"curiosity"|"user"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # traces: feedback, essais, etc.


class DagStore:
    """
    Store persistant pour un DAG d'objectifs.
    - Gère la création/màj de noeuds, liens, priorité.
    - Exporte un dashboard pour inspection runtime.
    """

    def __init__(self, persist_path: str = "data/goals.json", dashboard_path: str = "data/goals_dashboard.json"):
        self.persist_path = persist_path
        self.dashboard_path = dashboard_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.nodes: Dict[str, GoalNode] = {}
        self.active_goal_id: Optional[str] = None
        self._load()

    # ---------- CRUD ----------
    def add_goal(
        self,
        description: str,
        criteria: Optional[List[str]] = None,
        created_by: str = "system",
        value: float = 0.5,
        competence: float = 0.4,
        curiosity: float = 0.2,
        urgency: float = 0.3,
        parent_ids: Optional[List[str]] = None,
    ) -> GoalNode:
        gid = str(uuid.uuid4())[:8]
        node = GoalNode(
            id=gid,
            description=description,
            criteria=criteria or [],
            value=float(max(0, min(1, value))),
            competence=float(max(0, min(1, competence))),
            curiosity=float(max(0, min(1, curiosity))),
            urgency=float(max(0, min(1, urgency))),
            created_by=created_by,
            parent_ids=list(parent_ids or []),
        )
        self.nodes[gid] = node
        for p in node.parent_ids:
            if p in self.nodes and gid not in self.nodes[p].child_ids:
                self.nodes[p].child_ids.append(gid)
                self.nodes[p].updated_at = time.time()
        self._recompute_priority(node)
        self._save()
        self._export_dashboard()
        return node

    def link(self, parent_id: str, child_id: str):
        if parent_id in self.nodes and child_id in self.nodes:
            pa, ch = self.nodes[parent_id], self.nodes[child_id]
            if child_id not in pa.child_ids:
                pa.child_ids.append(child_id)
                pa.updated_at = time.time()
            if parent_id not in ch.parent_ids:
                ch.parent_ids.append(parent_id)
                ch.updated_at = time.time()
            self._save()
            self._export_dashboard()

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[GoalNode]:
        node = self.nodes.get(goal_id)
        if not node:
            return None
        for k, v in updates.items():
            if not hasattr(node, k):
                continue
            if isinstance(getattr(node, k), float):
                try:
                    v = float(v)
                except Exception:
                    pass
            setattr(node, k, v)
        node.updated_at = time.time()
        self._recompute_priority(node)
        self._save()
        self._export_dashboard()
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
            node.updated_at = time.time()
        self._recompute_priority(node)
        self._save()
        self._export_dashboard()
        return node

    def get_active(self) -> Optional[GoalNode]:
        if self.active_goal_id and self.active_goal_id in self.nodes:
            return self.nodes[self.active_goal_id]
        return None

    def complete_goal(self, goal_id: str, success: bool = True, note: str = ""):
        node = self.nodes.get(goal_id)
        if not node:
            return
        node.status = "done" if success else "abandoned"
        node.progress = 1.0 if success else node.progress
        node.updated_at = time.time()
        node.evidence.append({"t": time.time(), "type": "completion", "success": success, "note": note})
        if self.active_goal_id == goal_id:
            self.active_goal_id = None
        for ch in node.child_ids:
            if ch in self.nodes and self.nodes[ch].status == "blocked":
                self.nodes[ch].status = "pending"
        self._save()
        self._export_dashboard()

    # ---------- Priorité & tableaux ----------
    def recompute_all_priorities(self):
        for n in self.nodes.values():
            self._recompute_priority(n)
        self._save()
        self._export_dashboard()

    def topk(self, k: int = 5, only_pending: bool = True) -> List[GoalNode]:
        pool = [n for n in self.nodes.values() if (not only_pending) or n.status in ("pending", "active")]
        pool.sort(key=lambda n: n.priority, reverse=True)
        return pool[:k]

    # ---------- Helpers ----------
    def _info_gain_bonus(self, curiosity: float) -> float:
        return math.sqrt(max(0.0, curiosity)) * 0.6

    def _zpd_bonus(self, competence: float) -> float:
        return math.exp(-((competence - 0.5) ** 2) / (2 * 0.12))

    def _difficulty_shaping(self, competence: float) -> float:
        return max(0.0, 1.0 - abs(competence - 0.5) * 2.0)

    def _recompute_priority(self, node: GoalNode):
        ig = self._info_gain_bonus(node.curiosity)
        zpd = self._zpd_bonus(node.competence)
        diff = self._difficulty_shaping(node.competence)
        w_val, w_urg, w_ig, w_zpd, w_diff = 0.35, 0.25, 0.20, 0.10, 0.10
        raw = (
            (w_val * node.value)
            + (w_urg * node.urgency)
            + (w_ig * ig)
            + (w_zpd * zpd)
            + (w_diff * diff)
        )
        if node.status in ("blocked", "done", "abandoned"):
            raw *= 0.2
        node.priority = float(max(0.0, min(1.0, raw)))

    # ---------- Persistance ----------
    def _save(self):
        try:
            data = {
                "active_goal_id": self.active_goal_id,
                "nodes": {gid: asdict(n) for gid, n in self.nodes.items()},
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_goal_id = data.get("active_goal_id")
            self.nodes = {gid: GoalNode(**n) for gid, n in data.get("nodes", {}).items()}
        except Exception:
            self.nodes = {}
            self.active_goal_id = None

    def _export_dashboard(self):
        try:
            act = self.get_active()
            top = [asdict(n) for n in self.topk(10, only_pending=False)]
            payload = {
                "t": time.time(),
                "active_goal": asdict(act) if act else None,
                "top10": top,
                "counts": {
                    "total": len(self.nodes),
                    "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
                    "active": sum(1 for n in self.nodes.values() if n.status == "active"),
                    "done": sum(1 for n in self.nodes.values() if n.status == "done"),
                },
            }
            with open(self.dashboard_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
