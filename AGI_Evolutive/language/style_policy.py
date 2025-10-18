"""Language style policy definitions used by dialogue modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple


@dataclass
class StylePolicy:
    """Politique de style adaptable avec modes explicites."""

    params: Dict[str, float] = field(
        default_factory=lambda: {
            "politeness": 0.6,
            "directness": 0.5,
            "asking_rate": 0.4,
            "concretude_bias": 0.6,
            "hedging": 0.3,
            "warmth": 0.5,
            "verbosity": 0.6,
            "structure": 0.5,
        }
    )
    decay: float = 0.85
    current_mode: str = "pedagogique"
    persona_tone: str = "neutre"

    MODE_PRESETS: ClassVar[Dict[str, Dict[str, float]]] = {
        "brief": {
            "directness": 0.8,
            "verbosity": 0.3,
            "structure": 0.6,
            "asking_rate": 0.3,
        },
        "pedagogique": {
            "warmth": 0.7,
            "verbosity": 0.8,
            "structure": 0.75,
            "hedging": 0.35,
            "asking_rate": 0.5,
        },
        "audit": {
            "directness": 0.75,
            "hedging": 0.2,
            "asking_rate": 0.65,
            "structure": 0.8,
            "concretude_bias": 0.7,
        },
    }

    PERSONA_TONE_BIASES: ClassVar[Dict[str, Dict[str, float]]] = {
        "neutre": {},
        "chaleureux": {"warmth": 0.75, "politeness": 0.7},
        "professionnel": {"structure": 0.8, "directness": 0.6},
        "coach": {"warmth": 0.65, "asking_rate": 0.55},
    }

    MODE_KEYWORDS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "brief": ("mode bref", "mode synthétique", "sois bref", "fais court", "résume"),
        "pedagogique": ("mode pédagogique", "explique", "détaillé", "prends le temps"),
        "audit": ("mode audit", "challenge-moi", "questionne", "fais un audit"),
    }

    STYLE_MACROS: ClassVar[Dict[str, Dict[str, float]]] = {
        "taquin": {"warmth": +0.10, "directness": +0.10, "hedging": -0.10},
        "coach": {"warmth": +0.10, "asking_rate": +0.15},
        "sobre": {"structure": +0.10, "hedging": -0.05},
        "deadpan": {"warmth": -0.15, "structure": +0.10},
    }

    def set_mode(self, mode: str, persona_tone: Optional[str] = None) -> None:
        mode = (mode or "").lower()
        if mode not in self.MODE_PRESETS:
            return
        if persona_tone:
            self.persona_tone = persona_tone
        self.current_mode = mode
        self._apply_presets()

    def update_persona_tone(self, tone: str) -> None:
        if tone:
            self.persona_tone = tone.lower()
            self._apply_presets()

    def adapt_from_instruction(self, text: str) -> Dict[str, float]:
        """Infère des deltas de style depuis des instructions libres."""

        t = (text or "").lower()
        hints: Dict[str, float] = {}

        maybe_mode = self._detect_mode_keyword(t)
        if maybe_mode:
            self.set_mode(maybe_mode, persona_tone=self.persona_tone)

        warm_cues = ["bienveill", "empath", "chaleur", "gentil", "compréhens", "soigneux", "attentionné"]
        if any(c in t for c in warm_cues):
            hints["warmth"] = self._clip(self.params.get("warmth", 0.5) + 0.2)

        if any(k in t for k in ["prudent", "nuancé", "mesuré", "avec précaution"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) + 0.2)
        if any(k in t for k in ["direct", "cash", "franc", "sans détour", "tranché"]):
            hints["hedging"] = self._clip(self.params.get("hedging", 0.3) - 0.2)

        if any(k in t for k in ["concret", "exemples", "pratique", "spécifique"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) + 0.2)
        if any(k in t for k in ["haut niveau", "vue d'ensemble", "général"]):
            hints["concretude_bias"] = self._clip(self.params.get("concretude_bias", 0.6) - 0.2)

        if any(k in t for k in ["pose des questions", "questionne", "clarifie"]):
            hints["asking_rate"] = self._clip(self.params.get("asking_rate", 0.4) + 0.2)

        if any(k in t for k in ["structure", "plan", "étapes"]):
            hints["structure"] = self._clip(self.params.get("structure", 0.5) + 0.15)

        return hints

    def as_dict(self) -> Dict[str, float]:
        payload = dict(self.params)
        payload["mode"] = self.current_mode
        payload["persona_tone"] = self.persona_tone
        return payload

    def update_from_reward(self, reward: float) -> None:
        reward = float(max(-1.0, min(1.0, reward)))
        for key, coeff in {
            "concretude_bias": 0.1,
            "hedging": -0.05,
            "warmth": 0.08,
            "politeness": 0.06,
            "directness": 0.05,
        }.items():
            self.params[key] = self._clip(self.params.get(key, 0.5) + coeff * reward)

    def detect_mode_command(self, text: str) -> Optional[str]:
        """Détecte une commande explicite de changement de mode."""

        t = (text or "").strip().lower()
        if not t:
            return None
        if t.startswith("/mode"):
            parts = t.split()
            if len(parts) >= 2:
                candidate = parts[1]
                return candidate if candidate in self.MODE_PRESETS else None
        return self._detect_mode_keyword(t)

    def _detect_mode_keyword(self, text: str) -> Optional[str]:
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return mode
        return None

    def _apply_presets(self) -> None:
        base = {
            "politeness": 0.6,
            "directness": 0.5,
            "asking_rate": 0.4,
            "concretude_bias": 0.6,
            "hedging": 0.3,
            "warmth": 0.5,
            "verbosity": 0.6,
            "structure": 0.5,
        }
        mode_overrides = self.MODE_PRESETS.get(self.current_mode, {})
        persona_overrides = self.PERSONA_TONE_BIASES.get(self.persona_tone, {})
        for key, value in base.items():
            self.params[key] = value
        for overrides in (persona_overrides, mode_overrides):
            for key, value in overrides.items():
                self.params[key] = self._clip(value)

    def apply_macro(self, name: str) -> None:
        for key, delta in self.STYLE_MACROS.get(name, {}).items():
            base = self.params.get(key, 0.5)
            self.params[key] = self._clip(base + delta)

    @staticmethod
    def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))
