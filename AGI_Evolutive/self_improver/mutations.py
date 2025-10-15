from __future__ import annotations

import random
from typing import Any, Dict, List


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _mut(value: float, amp: float = 0.1) -> float:
    return _clip(value + random.uniform(-amp, amp))


def generate_overrides(base: Dict[str, Any], n: int = 4) -> List[Dict[str, Any]]:
    """Generate safe override candidates via lightweight mutations."""
    defaults = {
        "style.hedging": 0.3,
        "learning.self_assess.threshold": 0.90,
        "abduction.tie_gap": 0.12,
        "abduction.weights.prior": 0.5,
        "abduction.weights.boost": 1.0,
        "abduction.weights.match": 1.0,
    }

    base = dict(defaults, **(base or {}))
    keys = list(base.keys())
    candidates: List[Dict[str, Any]] = []

    if not keys:
        return [dict(defaults) for _ in range(max(1, n))]

    for _ in range(max(1, n)):
        candidate = dict(base)
        key = random.choice(keys)
        value = candidate[key]
        if isinstance(value, (int, float)):
            amp = 0.05 if "threshold" in key else 0.1
            candidate[key] = _mut(float(value), amp=amp)
        candidates.append(candidate)

    return candidates
