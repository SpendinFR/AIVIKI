from typing import List, Dict, Any, Optional
import random


class CuriosityEngine:
    """
    Sélectionne/crée des sous-buts alignés sur:
    - gain d'information attendu (curiosity),
    - zone proximale de développement (competence ~ 0.4–0.6),
    - lacunes détectées (métacognition / reasoning).
    """

    def __init__(self, architecture=None):
        self.arch = architecture

    # ---- Public API ----
    def suggest_subgoals(self, parent_goal: Optional[Dict[str, Any]] = None, k: int = 3) -> List[Dict[str, Any]]:
        """Retourne des propositions de sous-buts (dictionnaires pour DagStore.add_goal)."""
        signals = self._collect_signals()
        gaps = self._detect_gaps(signals)
        proposals = self._craft_goal_candidates(gaps, parent_goal, top_k=k)
        return proposals

    # ---- Internals ----
    def _collect_signals(self) -> Dict[str, Any]:
        m = getattr(self.arch, "metacognition", None)
        r = getattr(self.arch, "reasoning", None)

        perf = {}
        if m and hasattr(m, "get_metacognitive_status"):
            try:
                st = m.get_metacognitive_status()
                perf = st.get("performance_metrics", {})
            except Exception:
                perf = {}

        rstats = {}
        if r and hasattr(r, "get_reasoning_stats"):
            try:
                rstats = r.get_reasoning_stats()
            except Exception:
                rstats = {}

        return {"perf": perf, "reasoning": rstats}

    def _detect_gaps(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        gaps = []

        perf = signals.get("perf", {})
        low_perf = [(k, v) for k, v in perf.items() if v < 0.45]
        low_perf.sort(key=lambda kv: kv[1])

        for k, v in low_perf[:5]:
            gaps.append(
                {
                    "domain": k,
                    "severity": float(1.0 - v),
                    "hint": f"Baisser {k} (score={v:.2f}) suggère un manque de compétence/connaissance.",
                }
            )

        rs = signals.get("reasoning", {})
        common_errs = rs.get("common_errors", [])
        for e in common_errs:
            gaps.append({"domain": "reasoning_error", "severity": 0.6, "hint": f"Erreur récurrente: {e}"})

        return gaps

    def _craft_goal_candidates(
        self, gaps: List[Dict[str, Any]], parent_goal: Optional[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        proposals = []
        parent_id = parent_goal.get("id") if parent_goal else None

        for g in gaps[:5]:
            desc = self._gap_to_goal_description(g)
            proposals.append(
                {
                    "description": desc,
                    "criteria": self._default_criteria_for_gap(g),
                    "created_by": "curiosity",
                    "value": 0.45 + 0.15 * random.random(),
                    "competence": min(0.7, max(0.3, 0.45 + (random.random() - 0.5) * 0.2)),
                    "curiosity": min(1.0, 0.7 + 0.3 * random.random()),
                    "urgency": 0.3 + 0.2 * random.random(),
                    "parent_ids": [parent_id] if parent_id else [],
                }
            )

        if len(proposals) < top_k:
            proposals.append(
                {
                    "description": "Exploration guidée: cartographier un concept mal compris (auto-choix).",
                    "criteria": ["Produire un schéma/notes auto-explicatives."],
                    "created_by": "curiosity",
                    "value": 0.5,
                    "competence": 0.5,
                    "curiosity": 0.8,
                    "urgency": 0.3,
                    "parent_ids": [parent_id] if parent_id else [],
                }
            )

        random.shuffle(proposals)
        return proposals[:top_k]

    # ---- helpers ----
    def _gap_to_goal_description(self, gap: Dict[str, Any]) -> str:
        dom = gap.get("domain")
        if dom == "reasoning_speed":
            return "Améliorer la vitesse de raisonnement via micro-exercices chronométrés."
        if dom == "learning_rate":
            return "Augmenter le taux d’apprentissage: tester une nouvelle stratégie et mesurer."
        if dom == "recall_accuracy":
            return "Renforcer la mémoire de rappel: mettre en place répétition espacée ciblée."
        if dom == "reasoning_error":
            return "Réduire une erreur de raisonnement récurrente via contre-exemples."
        return f"Explorer et combler une lacune: {dom}"

    def _default_criteria_for_gap(self, gap: Dict[str, Any]) -> List[str]:
        dom = gap.get("domain")
        if dom == "reasoning_speed":
            return ["Atteindre un temps moyen < 0.7s sur 10 inférences tests"]
        if dom == "learning_rate":
            return ["Montrer +0.1 d’amélioration sur la métrique learning_rate en 24h"]
        if dom == "recall_accuracy":
            return ["Atteindre recall_accuracy ≥ 0.7 sur 20 items ciblés"]
        if dom == "reasoning_error":
            return ["Diminuer la fréquence de l’erreur ciblée de 50% sur 3 sessions"]
        return ["Définir un critère mesurable et l’atteindre (auto-évaluation)"]
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
