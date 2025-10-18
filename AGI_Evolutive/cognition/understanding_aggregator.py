from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UnderstandingResult:
    U_topic: float
    U_global: float
    details: Dict[str, float]


class UnderstandingAggregator:
    """Aggregates signals into understanding scores per topic and globally."""

    def __init__(self, alpha_topic: float = 0.25, alpha_global: float = 0.15) -> None:
        self.alpha_topic = alpha_topic
        self.alpha_global = alpha_global
        self._topic_ema: Dict[str, float] = {}
        self._global_ema: float = 0.5

    @staticmethod
    def _ema(prev: float, x: float, alpha: float) -> float:
        return (1 - alpha) * prev + alpha * x

    def compute(
        self,
        topic: Optional[str],
        prediction_error: float,
        memory_consistency: float = 0.5,
        transfer_success: float = 0.5,
        explanatory_adequacy: float = 0.5,
        social_appraisal: float = 0.5,
        clarification_penalty: float = 0.0,
        calibration_gap: float = 0.0,
    ) -> UnderstandingResult:
        pe = max(0.0, min(1.0, prediction_error))
        mc = max(0.0, min(1.0, memory_consistency))
        tr = max(0.0, min(1.0, transfer_success))
        ex = max(0.0, min(1.0, explanatory_adequacy))
        sa = max(0.0, min(1.0, social_appraisal))
        cp = max(0.0, min(1.0, clarification_penalty))
        cg = max(0.0, min(1.0, calibration_gap))

        U_inst = 0.40 * (1.0 - pe)
        U_inst += 0.15 * mc
        U_inst += 0.15 * tr
        U_inst += 0.15 * ex
        U_inst += 0.10 * sa
        U_inst -= 0.05 * cp
        U_inst -= 0.05 * cg
        U_inst = max(0.0, min(1.0, U_inst))

        if topic:
            prev_topic = self._topic_ema.get(topic, 0.5)
            U_topic = self._ema(prev_topic, U_inst, self.alpha_topic)
            self._topic_ema[topic] = U_topic
        else:
            U_topic = U_inst

        base = U_topic if topic else U_inst
        self._global_ema = self._ema(self._global_ema, base, self.alpha_global)

        return UnderstandingResult(
            U_topic=U_topic,
            U_global=self._global_ema,
            details={
                "prediction_error": pe,
                "memory_consistency": mc,
                "transfer_success": tr,
                "explanatory_adequacy": ex,
                "social_appraisal": sa,
                "clarification_penalty": cp,
                "calibration_gap": cg,
                "U_instant": U_inst,
            },
        )
