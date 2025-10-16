from __future__ import annotations

from typing import Dict

POSITIVE_MARKERS = {
    "merci": 0.25,
    "thanks": 0.25,
    "awesome": 0.25,
    "great": 0.25,
    "parfait": 0.25,
    "bravo": 0.25,
    "bien": 0.2,
    "excellent": 0.25,
    "top": 0.2,
    "génial": 0.25,
    "super": 0.25,
}

NEGATIVE_MARKERS = {
    "nul": -0.3,
    "bad": -0.3,
    "horrible": -0.3,
    "déçu": -0.25,
    "angry": -0.3,
    "non": -0.2,
    "pas clair": -0.25,
    "mauvais": -0.25,
    "bof": -0.2,
    "trop flou": -0.25,
    "à côté": -0.25,
}


def extract_social_reward(text: str) -> Dict[str, float]:
    """Évalue grossièrement la valence sociale d'un texte utilisateur."""

    lowered = (text or "").lower()
    score = 0.0

    for marker, weight in POSITIVE_MARKERS.items():
        if marker in lowered:
            score += weight

    for marker, weight in NEGATIVE_MARKERS.items():
        if marker in lowered:
            score += weight

    score = max(-1.0, min(1.0, score))
    return {"reward": float(score)}
