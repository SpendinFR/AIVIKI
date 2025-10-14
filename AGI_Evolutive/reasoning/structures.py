from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


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
    cost_est: float = 0.2
    expected_information_gain: float = 0.3


@dataclass
class Evidence:
    notes: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"notes": self.notes, "confidence": float(self.confidence)}
    confidence: float = 0.5


@dataclass
class Update:
    posterior: float
    decision: str


def now() -> float:
    return time.time()


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
    evidence: Optional[Evidence],
    result_text: str,
    final_confidence: float,
) -> Dict[str, Any]:
    return {
        "t": now(),
        "user_msg": user_msg,
        "hypotheses": [asdict(h) for h in hypotheses],
        "chosen_idx": chosen,
        "tests": [asdict(t) for t in tests],
        "evidence": asdict(evidence) if evidence else None,
        "solution": result_text,
        "final_confidence": final_confidence,
    }
