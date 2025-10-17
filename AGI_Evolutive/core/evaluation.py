def unified_priority(impact: float, probability: float, reversibility: float, effort: float,
                     uncertainty: float = 0.0, valence: float = 0.0) -> float:
    base = (max(0.0, impact) * max(0.0, probability) * max(0.0, reversibility)) / max(0.2, effort)
    mod = (1.0 - 0.5 * max(0.0, min(1.0, uncertainty))) * (1.0 + 0.3 * max(-1.0, min(1.0, valence)))
    return max(0.0, min(1.0, base * mod))
