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
