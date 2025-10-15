import json
import os
import time
from typing import Any, Dict

from AGI_Evolutive.core.config import cfg

_PATH = cfg()["HOMEOSTASIS_PATH"]


class Homeostasis:
    """Internal drives and reward shaping."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "drives": {
                "curiosity": 0.6,
                "self_preservation": 0.7,
                "social_bonding": 0.4,
                "competence": 0.5,
                "play": 0.3,
            },
            "last_update": time.time(),
            "intrinsic_reward": 0.0,
            "extrinsic_reward": 0.0,
        }
        self._load()

    def _load(self) -> None:
        if os.path.exists(_PATH):
            try:
                with open(_PATH, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def decay(self) -> None:
        for key, value in self.state["drives"].items():
            self.state["drives"][key] = value + 0.02 * (0.5 - value)

    def compute_intrinsic_reward(self, info_gain: float, progress: float) -> float:
        curiosity = self.state["drives"]["curiosity"]
        competence = self.state["drives"]["competence"]
        reward = 0.6 * curiosity * info_gain + 0.4 * competence * progress
        self.state["intrinsic_reward"] = reward
        self._save()
        return reward

    def compute_extrinsic_reward_from_memories(self, recent_feedback: str) -> float:
        text = (recent_feedback or "").lower()
        bonus = 0.0
        if any(word in text for word in ["bravo", "bien", "good", "thanks", "merci"]):
            bonus += 0.3
        if any(word in text for word in ["mauvais", "non", "wrong", "bad"]):
            bonus -= 0.2
        self.state["extrinsic_reward"] = max(-1.0, min(1.0, bonus))
        self._save()
        return self.state["extrinsic_reward"]

    def update_from_rewards(self, intrinsic: float, extrinsic: float) -> None:
        self.state["drives"]["curiosity"] = max(
            0, min(1, self.state["drives"]["curiosity"] + 0.05 * (intrinsic - 0.5))
        )
        self.state["drives"]["competence"] = max(
            0, min(1, self.state["drives"]["competence"] + 0.05 * (intrinsic - 0.5))
        )
        self.state["drives"]["social_bonding"] = max(
            0, min(1, self.state["drives"]["social_bonding"] + 0.05 * extrinsic)
        )
        self._save()
