"""Lightweight data structures used by the reasoning module."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Hypothesis:
    content: str
    prior: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["prior"] = float(self.prior)
        return data


@dataclass
class Test:
    description: str
    cost_est: float = 0.2
    expected_information_gain: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "cost_est": float(self.cost_est),
            "expected_information_gain": float(self.expected_information_gain),
        }


@dataclass
class Evidence:
    notes: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"notes": self.notes, "confidence": float(self.confidence)}


@dataclass
class Update:
    posterior: float
    decision: str
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["posterior"] = float(self.posterior)
        return payload


def now() -> float:
    return time.time()


def episode_record(
    user_msg: str,
    hypotheses: List[Hypothesis],
    chosen_index: int,
    tests: List[Test],
    evidence: Evidence,
    result_text: str,
    final_confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "timestamp": now(),
        "user_msg": user_msg,
        "hypotheses": [h.to_dict() for h in hypotheses],
        "chosen_index": int(chosen_index),
        "tests": [t.to_dict() for t in tests],
        "evidence": evidence.to_dict(),
        "result_text": result_text,
        "final_confidence": float(final_confidence),
    }
    if metadata:
        record.update(metadata)
    return record


def summary_record(
    user_msg: str,
    hypotheses: List[Hypothesis],
    chosen_index: int,
    tests: List[Test],
    evidence: Optional[Evidence],
    result_text: str,
    final_confidence: float,
) -> Dict[str, Any]:
    return {
        "t": now(),
        "user_msg": user_msg,
        "hypotheses": [h.to_dict() for h in hypotheses],
        "chosen_idx": int(chosen_index),
        "tests": [t.to_dict() for t in tests],
        "evidence": evidence.to_dict() if evidence else None,
        "solution": result_text,
        "final_confidence": float(final_confidence),
    }
