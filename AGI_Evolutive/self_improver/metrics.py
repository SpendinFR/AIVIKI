from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List
import random


def _iter_metric_keys(samples: Iterable[Dict[str, Any]]) -> List[str]:
    keys = set()
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def aggregate_metrics(samples: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metric samples by averaging shared numeric keys."""
    if not samples:
        return {"acc": 0.0, "cal_ece": 1.0, "time": 0.0}

    keys = _iter_metric_keys(samples)
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if values:
            aggregated[key] = float(mean(values))
    return aggregated


def dominates(
    champion: Dict[str, float],
    challenger: Dict[str, float],
    eps_acc: float = 0.01,
    max_ece_worsen: float = 0.02,
    max_time_increase: float = 0.15,
) -> bool:
    """Return True if the challenger dominates the champion under the given thresholds."""
    acc_c = challenger.get("acc", 0.0)
    acc_p = champion.get("acc", 0.0)
    ece_c = challenger.get("cal_ece", 1.0)
    ece_p = champion.get("cal_ece", 1.0)
    t_c = challenger.get("time", 1.0)
    t_p = champion.get("time", 1.0)

    if acc_c < acc_p + eps_acc:
        return False
    if (ece_c - ece_p) > max_ece_worsen:
        return False
    if (t_c - t_p) > max_time_increase * max(1e-6, t_p):
        return False
    return True


def bootstrap_superiority(
    champion_scores: List[float],
    challenger_scores: List[float],
    trials: int = 1000,
) -> float:
    """Approximate one-sided p-value via simple bootstrap sampling."""
    if not champion_scores or not challenger_scores:
        return 1.0

    wins = 0
    for _ in range(max(1, trials)):
        a = random.choice(champion_scores)
        b = random.choice(challenger_scores)
        if b > a:
            wins += 1
    return 1.0 - (wins / max(1, trials))
