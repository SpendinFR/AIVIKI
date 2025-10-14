from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import os
import json
from .dag_store import GoalDAG, GoalNode


def _now():
    return time.time()


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"last_seen": {}, "attempts": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_seen": {}, "attempts": {}}


def _write_history(path: str, d: Dict[str, Any]):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _novelty(last_ts: Optional[float], horizon: float = 300.0) -> float:
    """
    + élevé si le nœud n'a pas été travaillé récemment.
    horizon=300s → après 5 min sans travail, novelty≈1.0
    """
    if not last_ts:
        return 1.0
    dt = max(0.0, _now() - last_ts)
    return float(max(0.0, min(1.0, dt / horizon)))


def estimate_information_gain(node: GoalNode, last_ts: Optional[float]) -> float:
    uncertainty = 1.0 - node.competence
    novelty = _novelty(last_ts)
    ig = 0.6 * uncertainty + 0.4 * novelty
    return float(max(0.0, min(1.0, ig)))


def select_next_subgoals(
    dag: GoalDAG,
    k: int = 3,
    history_path: str = "logs/curiosity_history.json",
) -> List[Tuple[GoalNode, float]]:
    """
    Retourne jusqu'à k sous-buts prometteurs (node, score).
    Score combine: information_gain, ZPD (via priority du DAG), incompletion.
    """
    hist = _read_history(history_path)
    frontier = dag.frontier()
    scored: List[Tuple[GoalNode, float]] = []

    for node in frontier:
        last_ts = hist.get("last_seen", {}).get(node.goal_id)
        ig = estimate_information_gain(node, last_ts)
        score = dag.compute_priority(node.goal_id, curiosity_score=ig)
        scored.append((node, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = scored[:k]

    for n, _ in chosen:
        hist.setdefault("last_seen", {})[n.goal_id] = _now()
        hist.setdefault("attempts", {})[n.goal_id] = hist["attempts"].get(n.goal_id, 0) + 1

    _write_history(history_path, hist)
    return chosen
