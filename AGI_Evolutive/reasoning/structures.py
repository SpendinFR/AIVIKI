import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Hypothesis:
    content: str
    prior: float = 0.5


@dataclass
class Test:
    description: str
    cost_est: float = 0.2
    expected_information_gain: float = 0.3


@dataclass
class Evidence:
    notes: str
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
