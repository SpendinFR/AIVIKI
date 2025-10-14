from __future__ import annotations

from typing import Dict

POSITIVE_MARKERS = {"merci", "thanks", "awesome", "great", "parfait"}
NEGATIVE_MARKERS = {"nul", "bad", "horrible", "déçu", "angry"}


def extract_social_reward(text: str) -> Dict[str, float]:
    text_lower = (text or "").lower()
    reward = 0.0
    for marker in POSITIVE_MARKERS:
        if marker in text_lower:
            reward += 0.2
    for marker in NEGATIVE_MARKERS:
        if marker in text_lower:
            reward -= 0.3
    return {"reward": float(max(-1.0, min(1.0, reward)))}
from typing import Dict

POSITIVE_CUES = ["bravo", "bien", "parfait", "excellent", "merci", "top", "génial", "super"]
NEGATIVE_CUES = ["non", "pas clair", "mauvais", "nul", "bof", "trop flou", "à côté"]


def extract_social_reward(text: str) -> Dict[str, float]:
    """
    Détecte un signal social simple (louange / critique).
    Retourne un dict {valence:intensité in [0,1]} ; signe par valence.
    """
    t = (text or "").lower()
    score = 0.0
    for w in POSITIVE_CUES:
        if w in t:
            score += 0.25
    for w in NEGATIVE_CUES:
        if w in t:
            score -= 0.25
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0
    return {"reward": score}
