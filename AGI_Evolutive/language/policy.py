from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StylePolicy:
    """Petite politique de style adaptative basée sur un signal social cumulatif."""

    params: Dict[str, float] = field(
        default_factory=lambda: {
            "politeness": 0.6,
            "directness": 0.5,
            "asking_rate": 0.4,
        }
    )
    decay: float = 0.85

    def update_from_reward(self, reward: float) -> None:
        """Ajuste légèrement les paramètres selon la récompense sociale."""
        adj = max(min(reward, 1.0), -1.0)
        self.params["politeness"] = self._clip(
            self.decay * self.params["politeness"] + 0.1 * adj + 0.5 * (1 - self.decay)
        )
        self.params["directness"] = self._clip(
            self.decay * self.params["directness"] + 0.15 * adj + 0.5 * (1 - self.decay)
        )
        self.params["asking_rate"] = self._clip(
            self.decay * self.params["asking_rate"] + 0.2 * adj + 0.4 * (1 - self.decay)
        )

    def as_dict(self) -> Dict[str, float]:
        return dict(self.params)

    @staticmethod
    def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))
