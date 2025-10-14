from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class Hypothesis:
    content: str
    prior: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["content"] = self.content
        data["prior"] = float(self.prior)
        return data


@dataclass
class Test:
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {"description": self.description}


@dataclass
class Evidence:
    notes: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"notes": self.notes, "confidence": float(self.confidence)}


def episode_record(
    user_msg: str,
    hypotheses: List[Hypothesis],
    chosen: int,
    tests: List[Test],
    evidence: Evidence,
    result_text: str,
    final_confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "timestamp": time.time(),
        "user_msg": user_msg,
        "hypotheses": [h.to_dict() for h in hypotheses],
        "chosen_index": int(chosen),
        "tests": [t.to_dict() for t in tests],
        "evidence": evidence.to_dict(),
        "result_text": result_text,
        "final_confidence": float(final_confidence),
    }
    if metadata:
        record.update(metadata)
    return record
