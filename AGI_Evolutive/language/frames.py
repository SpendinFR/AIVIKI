from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto


class DialogueAct(Enum):
    ASK = auto()
    INFORM = auto()
    REQUEST = auto()
    FEEDBACK_POS = auto()
    FEEDBACK_NEG = auto()
    GREET = auto()
    BYE = auto()
    THANKS = auto()
    META_HELP = auto()
    CLARIFY = auto()
    REFLECT = auto()


@dataclass
class UtteranceFrame:
    text: str
    normalized_text: str
    intent: str
    confidence: float
    uncertainty: float
    acts: List[DialogueAct] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    unknown_terms: List[str] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)      # ce dont l’IA a “besoin” pour bien répondre
    meta: Dict[str, Any] = field(default_factory=dict)  # ex: language, tone, user_profile hints

    @property
    def surface_form(self) -> str:
        """Compat: utilisé par ton ancien cycle."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "acts": [a.name for a in self.acts],
            "slots": self.slots,
            "unknown_terms": self.unknown_terms,
            "needs": self.needs,
            "meta": self.meta,
        }
