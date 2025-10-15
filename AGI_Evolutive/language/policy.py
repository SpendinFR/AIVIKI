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

    def adapt_from_instruction(self, text: str) -> Dict[str, float]:
        """
        Infère des deltas de style depuis une instruction en langage naturel.
        Retourne des overrides pour *le tour courant* (pas de persistance ici).
        """
        t = (text or "").lower()
        hints: Dict[str, float] = {}

        # Chaleur / bienveillance
        warm_cues = ["bienveill", "empath", "chaleur", "gentil", "compréhens", "soigneux", "attentionné"]
        if any(c in t for c in warm_cues):
            hints["warmth"] = self._clip(self.params.get("warmth", 0.5) + 0.2)

        # Prudence (hedging) up/down
        if any(k in t for k in ["prudent", "nuancé", "mesuré", "avec précaution"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) + 0.2)
        if any(k in t for k in ["direct", "cash", "franc", "sans détour", "tranché"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) - 0.2)

        # Concrétude / exemples
        if any(k in t for k in ["concret", "exemples", "pratique", "spécifique"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) + 0.2)
        if any(k in t for k in ["haut niveau", "vue d'ensemble", "général"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) - 0.2)

        # Propension à questionner
        if any(k in t for k in ["pose des questions", "questionne", "clarifie"]):
            hints["asking_rate"] = self._clip(self.params.get("asking_rate", 0.2) + 0.2)

        return hints

    def as_dict(self) -> Dict[str, float]:
        return dict(self.params)

    def update_from_reward(self, reward: float) -> None:
        reward = float(max(-1.0, min(1.0, reward)))
        self.params["concretude_bias"] = min(1.0, max(0.0, self.params["concretude_bias"] + 0.1 * reward))
        self.params["hedging"] = min(1.0, max(0.0, self.params["hedging"] - 0.05 * reward))
        self.params["warmth"] = min(1.0, max(0.0, self.params["warmth"] + 0.08 * reward))
    @staticmethod
    def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))
