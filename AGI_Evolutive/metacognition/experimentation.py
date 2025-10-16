from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass, asdict
from collections import deque
import json
import os
import time
import uuid

from AGI_Evolutive.utils.jsonsafe import json_sanitize


def _now():
    return time.time()


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


EXPERIMENTS_LOG = "logs/experiments.jsonl"


@dataclass
class Experiment:
    exp_id: str
    metric: str
    baseline: float
    target_change: float
    plan: Dict[str, Any]
    duration_cycles: int = 3
    created_at: float = 0.0
    status: str = "scheduled"
    notes: str = ""

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(json_sanitize(d), ensure_ascii=False)


class MetacognitionExperimenter:
    """
    Génère de petits tests concrets à partir des métriques,
    et consigne le résultat pour fermer la boucle.
    """

    def __init__(self, system_ref: Any = None):
        self.system = system_ref
        self.active: deque[Experiment] = deque(maxlen=50)

    def suggest_and_log_tests(self, performance_metrics: Dict[str, float]):
        """
        Pour chaque métrique clé, si sous un seuil → proposer un test.
        """
        candidates = {
            "reasoning_speed": {
                "improve": +0.12,
                "plan": {
                    "strategy": "focused_cycles",
                    "details": "2 cycles attention_focused puis 1 cycle normal",
                },
            },
            "learning_rate": {
                "improve": +0.10,
                "plan": {
                    "strategy": "spaced_review",
                    "details": "répéter 2 prompts similaires à 1 min d'intervalle",
                },
            },
            "recall_accuracy": {
                "improve": +0.08,
                "plan": {
                    "strategy": "elaborative_q",
                    "details": "ajouter 2 'pourquoi' à chaque explication",
                },
            },
        }

        created = 0
        for m, cfg in candidates.items():
            if m not in performance_metrics:
                continue
            baseline = performance_metrics[m]
            if baseline < 0.7:
                exp = Experiment(
                    exp_id=str(uuid.uuid4())[:8],
                    metric=m,
                    baseline=float(baseline),
                    target_change=float(cfg["improve"]),
                    plan=cfg["plan"],
                    duration_cycles=3,
                    created_at=_now(),
                    status="scheduled",
                )
                self._append_jsonl(EXPERIMENTS_LOG, exp.to_jsonl())
                self.active.append(exp)
                created += 1

        if created and self.system:
            try:
                self.system._record_metacognitive_event(
                    event_type="experiment_planned",
                    domain=self.system.CognitiveDomain.LEARNING
                    if hasattr(self.system, "CognitiveDomain")
                    else None,
                    description=f"{created} test(s) planifié(s) pour optimisation métriques.",
                    significance=min(0.3 + 0.1 * created, 0.7),
                    confidence=0.6,
                )
            except Exception:
                pass

    def record_outcome(self, metric: str, new_value: float):
        """
        Enregistre un résultat pour un test ciblant 'metric' (si actif).
        """
        exp = None
        for e in reversed(self.active):
            if e.metric == metric and e.status in ("scheduled", "running"):
                exp = e
                break
        if not exp:
            return

        goal = exp.baseline * (1.0 + exp.target_change)
        success = bool(new_value >= goal)

        outcome = {
            "exp_id": exp.exp_id,
            "metric": metric,
            "baseline": exp.baseline,
            "observed": float(new_value),
            "goal": goal,
            "success": success,
            "measured_at": _now(),
        }
        self._append_jsonl(
            EXPERIMENTS_LOG,
            json.dumps(json_sanitize({"outcome": outcome}), ensure_ascii=False),
        )
        exp.status = "done" if success else "failed"

        if self.system:
            try:
                self.system._record_metacognitive_event(
                    event_type="experiment_result",
                    domain=self.system.CognitiveDomain.LEARNING
                    if hasattr(self.system, "CognitiveDomain")
                    else None,
                    description=(
                        f"Résultat test {metric}: {'OK' if success else 'KO'} "
                        f"(observé={new_value:.2f}, cible={goal:.2f})"
                    ),
                    significance=0.5 if success else 0.3,
                    confidence=0.7,
                )
            except Exception:
                pass

    @staticmethod
    def _append_jsonl(path: str, line: str):
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


ABILITY_TO_METRIC = {
    "reasoning_speed": "reasoning_speed",
    "learning_efficiency": "learning_rate",
    "memory_capacity": "memory_capacity",
}


def calibrate_self_model(
    self_model: Any, performance_tracking: Dict[str, list], learning_rate: float = 0.1
) -> Dict[str, float]:
    """
    Compare auto-évaluations vs performances (dernière valeur),
    puis ajuste doucement les capacités du self_model.
    Retourne les deltas appliqués.
    """
    applied = {}
    for ability, metric in ABILITY_TO_METRIC.items():
        hist = performance_tracking.get(metric, [])
        if not hist:
            continue
        observed = float(hist[-1].get("value", 0.5))
        current = float(self_model.cognitive_abilities.get(ability, 0.5))
        new_val = (1 - learning_rate) * current + learning_rate * observed
        self_model.cognitive_abilities[ability] = max(0.0, min(1.0, new_val))
        applied[ability] = self_model.cognitive_abilities[ability] - current

    hist_load = performance_tracking.get("cognitive_load", [])
    if hist_load:
        observed_load = float(hist_load[-1].get("value", 0.5))
        observed_attention = 1.0 - observed_load
        current = float(self_model.cognitive_abilities.get("attention_control", 0.5))
        new_val = (1 - learning_rate) * current + learning_rate * observed_attention
        self_model.cognitive_abilities["attention_control"] = max(
            0.0, min(1.0, new_val)
        )
        applied["attention_control"] = (
            self_model.cognitive_abilities["attention_control"] - current
        )

    return applied
