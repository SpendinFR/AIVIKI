import os
import json
import time
from typing import Dict, Any

class Homeostasis:
    """Maintains simple drive levels and computes intrinsic/extrinsic rewards."""

    def __init__(self, path: str = "data/homeostasis.json"):
        self.path = path
        self.state: Dict[str, Any] = {
            "drives": {
                "curiosity": 0.6,
                "competence": 0.5,
                "social_bonding": 0.5,
                "safety": 0.7,
            },
            "last_update": 0.0
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def _decay(self):
        drives = self.state["drives"]
        for k, v in drives.items():
            drives[k] = max(0.0, min(1.0, 0.995 * v + 0.005 * 0.5))
        self.state["last_update"] = time.time()
        self._save()

    def compute_intrinsic_reward(self, info_gain: float, progress: float) -> float:
        self._decay()
        curiosity = self.state["drives"].get("curiosity", 0.5)
        competence = self.state["drives"].get("competence", 0.5)
        reward = 0.4 * curiosity * info_gain + 0.6 * competence * progress
        reward = max(0.0, min(1.0, reward))
        self.state.setdefault("metrics", {}).setdefault("intrinsic", []).append({
            "ts": time.time(),
            "value": reward,
            "info_gain": info_gain,
            "progress": progress
        })
        self.state["metrics"]["intrinsic"] = self.state["metrics"]["intrinsic"][-200:]
        self._save()
        return reward

    def compute_extrinsic_reward_from_memories(self, feedback_text: str) -> float:
        text = (feedback_text or "").lower()
        positive = sum(text.count(w) for w in ["merci", "bravo", "good", "bien"])
        negative = sum(text.count(w) for w in ["erreur", "mauvais", "fail", "triste"])
        base = 0.5 + 0.1 * (positive - negative)
        base = max(0.0, min(1.0, base))
        self.state.setdefault("metrics", {}).setdefault("extrinsic", []).append({
            "ts": time.time(),
            "value": base,
            "positive": positive,
            "negative": negative
        })
        self.state["metrics"]["extrinsic"] = self.state["metrics"]["extrinsic"][-200:]
        self._save()
        return base

    def adjust_drive(self, drive: str, delta: float):
        if drive not in self.state["drives"]:
            return
        self.state["drives"][drive] = max(0.0, min(1.0, self.state["drives"][drive] + delta))
        self._save()

    def set_drive_target(self, drive: str, target: float):
        if drive not in self.state["drives"]:
            return
        self.state["drives"][drive] = max(0.0, min(1.0, target))
        self._save()
