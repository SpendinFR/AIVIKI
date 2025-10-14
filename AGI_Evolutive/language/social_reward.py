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
