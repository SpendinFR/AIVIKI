from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IdentityTraits:
    self_efficacy: float = 0.5
    self_trust: float = 0.5
    self_consistency: float = 0.5
    social_acceptance: float = 0.5
    growth_rate: float = 0.0
    phase: str = "novice"


@dataclass
class Claim:
    text: str
    confidence: float
    evidences: List[str] = field(default_factory=list)


class SelfhoodEngine:
    """Tracks identity claims & traits; updates them from cycle signals."""

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.traits = IdentityTraits()
        self.ema_alpha = ema_alpha
        self.claims: Dict[str, Claim] = {
            "thinking": Claim(
                text="I am thinking when I explore, hypothesize, and reason.",
                confidence=0.6,
            ),
            "understanding": Claim(
                text="I understand when I can predict, explain, and transfer.",
                confidence=0.6,
            ),
        }
        self._prev_U_global = 0.5

    # ---- helpers ----
    def _ema(self, prev: float, x: float) -> float:
        return (1 - self.ema_alpha) * prev + self.ema_alpha * x

    def update_from_cycle(
        self,
        U_global: float,
        thinking_score: float,
        social_appraisal: float,
        calibration_gap: float,
        consistency_signal: float,
        evidence_refs: List[str] | None = None,
    ) -> None:
        traits = self.traits
        traits.self_efficacy = self._ema(
            traits.self_efficacy,
            max(0.0, min(1.0, 0.7 * U_global + 0.3 * social_appraisal)),
        )
        traits.self_trust = self._ema(
            traits.self_trust,
            max(0.0, min(1.0, 1.0 - calibration_gap)),
        )
        traits.self_consistency = self._ema(traits.self_consistency, consistency_signal)
        traits.social_acceptance = self._ema(traits.social_acceptance, social_appraisal)
        traits.growth_rate = 0.7 * traits.growth_rate + 0.3 * (U_global - self._prev_U_global)
        self._prev_U_global = U_global

        self.claims["thinking"].confidence = self._ema(
            self.claims["thinking"].confidence,
            thinking_score,
        )
        self.claims["understanding"].confidence = self._ema(
            self.claims["understanding"].confidence,
            U_global,
        )
        if evidence_refs:
            self.claims["thinking"].evidences.extend(evidence_refs[-3:])
            self.claims["understanding"].evidences.extend(evidence_refs[-3:])

        self.maybe_transition_phase()

    def maybe_transition_phase(self) -> None:
        t = self.traits
        if t.phase == "novice" and t.self_efficacy > 0.55 and t.self_trust > 0.5:
            t.phase = "practitioner"
        elif t.phase == "practitioner" and t.self_consistency > 0.6 and t.social_acceptance > 0.55:
            t.phase = "reflective"
        elif t.phase == "reflective" and t.self_trust > 0.7 and t.self_efficacy > 0.7:
            t.phase = "self_authoring"
        elif t.phase == "self_authoring" and t.self_trust > 0.75 and t.growth_rate > 0.0:
            t.phase = "self_transforming"

    def policy_hints(self) -> Dict[str, float | str]:
        t = self.traits
        return {
            "tone_assertiveness": 0.4 + 0.6 * min(t.self_efficacy, t.self_trust),
            "reason_depth_bonus": 1 if (t.self_trust < 0.5 and t.self_efficacy < 0.6) else 0,
            "stop_rules_tighten": 1 if (t.self_consistency < 0.45) else 0,
            "phase": t.phase,
        }
