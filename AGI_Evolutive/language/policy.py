from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StylePolicy:
    params: Dict[str, float] = field(default_factory=lambda: {
        "concretude_bias": 0.6,
        "hedging": 0.3,
        "warmth": 0.5,
    })

    def as_dict(self) -> Dict[str, float]:
        return dict(self.params)

    def update_from_reward(self, reward: float) -> None:
        reward = float(max(-1.0, min(1.0, reward)))
        self.params["concretude_bias"] = min(1.0, max(0.0, self.params["concretude_bias"] + 0.1 * reward))
        self.params["hedging"] = min(1.0, max(0.0, self.params["hedging"] - 0.05 * reward))
        self.params["warmth"] = min(1.0, max(0.0, self.params["warmth"] + 0.08 * reward))
