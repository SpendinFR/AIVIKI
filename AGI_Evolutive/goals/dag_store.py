from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import time
import json
import os


def _now() -> float:
    return time.time()


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class GoalNode:
    goal_id: str
    description: str
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    value: float = 0.5
    competence: float = 0.5
    status: str = "active"
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_worked: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoalNode":
        return GoalNode(**d)

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
import json
import os
from typing import Any, Dict, List, Optional

_DEFAULT_DAG = {
    "evolve": {
        "id": "evolve",
        "parents": [],
        "progress": 0.10,
        "values": {"curiosity": 0.8, "truth": 0.7},
    },
    "understand_env": {
        "id": "understand_env",
        "parents": ["evolve"],
        "progress": 0.20,
    },
    "understand_humans": {
        "id": "understand_humans",
        "parents": ["understand_env"],
        "progress": 0.18,
        "criteria": [
            "déduire intention",
            "ajuster style",
            "prédire reward social",
        ],
    },
    "tooling_mastery": {
        "id": "tooling_mastery",
        "parents": ["evolve"],
        "progress": 0.12,
    },
    "self_modeling": {
        "id": "self_modeling",
        "parents": ["evolve"],
        "progress": 0.14,
    },
}


class GoalDAG:
    """
    DAG d'objectifs léger (sans dépendances externes).
    - nodes: goal_id -> GoalNode
    - parents/children: adjacence
    - sauvegarde JSON (pas JSONL) pour état global.
    """

    def __init__(self):
        self.nodes: Dict[str, GoalNode] = {}
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.parents: Dict[str, Set[str]] = defaultdict(set)

    # -------------------- CRUD --------------------

    def add_goal(self, goal_id: str, **kwargs) -> GoalNode:
        if goal_id in self.nodes:
            node = self.nodes[goal_id]
            for k, v in kwargs.items():
                if hasattr(node, k) and v is not None:
                    setattr(node, k, v)
            node.updated_at = _now()
            return node

        node = GoalNode(goal_id=goal_id, **kwargs)
        self.nodes[goal_id] = node
        return node

    def add_subgoal(self, parent_id: str, child_id: str, **kwargs) -> GoalNode:
        child = self.add_goal(child_id, **kwargs)
        self.children[parent_id].add(child_id)
        self.parents[child_id].add(parent_id)
        return child

    def get_node(self, goal_id: str) -> Optional[GoalNode]:
        return self.nodes.get(goal_id)

    def mark_done(self, goal_id: str):
        node = self.nodes.get(goal_id)
        if not node:
            return
        node.progress = 1.0
        node.status = "done"
        node.updated_at = _now()
        self._propagate_progress_up(goal_id)

    def update_progress(
        self,
        goal_id: str,
        progress: float,
        competence_delta: Optional[float] = None,
    ):
        node = self.nodes.get(goal_id)
        if not node:
            return
        node.progress = float(max(0.0, min(1.0, progress)))
        if competence_delta is not None:
            node.competence = float(
                max(0.0, min(1.0, node.competence + competence_delta))
            )
        node.last_worked = _now()
        node.updated_at = _now()
        self._propagate_progress_up(goal_id)

    # -------------------- Requêtes --------------------

    def children_of(self, goal_id: str) -> List[GoalNode]:
        return [self.nodes[c] for c in self.children.get(goal_id, set())]

    def parents_of(self, goal_id: str) -> List[GoalNode]:
        return [self.nodes[p] for p in self.parents.get(goal_id, set())]

    def leaves(self) -> List[GoalNode]:
        return [
            n
            for gid, n in self.nodes.items()
            if len(self.children.get(gid, set())) == 0
        ]

    def frontier(self) -> List[GoalNode]:
        """
        Sous-buts "travaillables" maintenant :
        - actifs
        - non terminés
        - soit feuilles, soit parents avec au moins un enfant non terminé
        """
        out = []
        for gid, node in self.nodes.items():
            if node.status != "active" or node.progress >= 1.0:
                continue
            ch = self.children.get(gid, set())
            if not ch:
                out.append(node)
            else:
                if any(self.nodes[c].progress < 1.0 for c in ch):
                    out.append(node)
        return out

    # -------------------- Heuristiques --------------------

    def compute_priority(self, goal_id: str, curiosity_score: float = 0.0) -> float:
        """
        Score combiné :
        - utilité (value)
        - info/curiosity
        - incomplétude (1 - progress)
        - zone proximale d’apprentissage (boost si competence in [0.4,0.6])
        """
        n = self.nodes[goal_id]
        incompleteness = 1.0 - n.progress
        zpd = (
            1.2
            if 0.4 <= n.competence <= 0.6
            else (0.9 if (n.competence < 0.2 or n.competence > 0.8) else 1.0)
        )
        base = 0.5 * n.value + 0.5 * curiosity_score
        return float(max(0.0, min(1.0, base * incompleteness * zpd)))

    # -------------------- Sauvegarde --------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {gid: n.to_dict() for gid, n in self.nodes.items()},
            "children": {gid: list(ch) for gid, ch in self.children.items()},
            "parents": {gid: list(pa) for gid, pa in self.parents.items()},
            "saved_at": _now(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoalDAG":
        dag = GoalDAG()
        for gid, nd in d.get("nodes", {}).items():
            dag.nodes[gid] = GoalNode.from_dict(nd)
        for gid, lst in d.get("children", {}).items():
            dag.children[gid] = set(lst)
        for gid, lst in d.get("parents", {}).items():
            dag.parents[gid] = set(lst)
        return dag

    def save(self, path: str = "logs/goals_dag.json"):
        _ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str = "logs/goals_dag.json") -> "GoalDAG":
        if not os.path.exists(path):
            return GoalDAG()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return GoalDAG.from_dict(d)

    # -------------------- Internes --------------------

    def _propagate_progress_up(self, goal_id: str):
        for parent in self.parents.get(goal_id, set()):
            ch = list(self.children.get(parent, set()))
            if ch:
                avg = sum(self.nodes[c].progress for c in ch) / len(ch)
                pn = self.nodes[parent]
                pn.progress = avg
                pn.updated_at = _now()
                if avg >= 1.0:
                    pn.status = "done"
            self._propagate_progress_up(parent)
    DAG d’objectifs persistant + choix de sous-but par EVI simple.
    """

    def __init__(self, path: str = "runtime/goal_dag.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.graph = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        self._persist(_DEFAULT_DAG)
        return dict(_DEFAULT_DAG)

    def _persist(self, graph: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

    def get(self, goal_id: str) -> Optional[Dict[str, Any]]:
        return self.graph.get(goal_id)

    def bump_progress(self, goal_id: str, delta: float) -> float:
        if goal_id not in self.graph:
            return 0.0
        g = self.graph[goal_id]
        g["progress"] = max(0.0, min(1.0, g.get("progress", 0.0) + delta))
        self._persist(self.graph)
        return g["progress"]

    def rollup_progress(self, goal_id: str) -> float:
        """Propage la progression (moyenne des enfants si existants)."""
        children = [
            g
            for g in self.graph.values()
            if goal_id in g.get("parents", [])
        ]
        if not children:
            return self.graph.get(goal_id, {}).get("progress", 0.0)
        vals = [self.rollup_progress(ch["id"]) for ch in children]
        val = sum(vals) / len(vals)
        self.graph[goal_id]["progress"] = val
        return val

    def _simple_evi(self, goal: Dict[str, Any]) -> float:
        """
        Valeur d’info attendue naïve:
        - plus le goal est peu avancé, plus EVI ↑
        - petit bonus si lié à 'understand_humans' (interaction)
        """
        p = 1.0 - goal.get("progress", 0.0)
        bonus = 0.1 if goal["id"] in ("understand_humans", "self_modeling") else 0.0
        return max(0.0, min(1.0, 0.6 * p + bonus))

    def choose_next_goal(self) -> Dict[str, Any]:
        """Choisit un sous-but à forte EVI parmi les feuilles (ou quasi-feuilles)."""
        candidates: List[Dict[str, Any]] = []
        for g in self.graph.values():
            # feuille = pas d'enfant
            has_child = any(
                g["id"] in other.get("parents", []) for other in self.graph.values()
            )
            if not has_child:
                candidates.append(g)
        if not candidates:
            candidates = list(self.graph.values())
        ranked = sorted(
            candidates, key=lambda gg: self._simple_evi(gg), reverse=True
        )
        top = ranked[0]
        return {
            "id": top["id"],
            "evi": self._simple_evi(top),
            "progress": top.get("progress", 0.0),
        }
