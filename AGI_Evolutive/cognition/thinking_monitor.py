from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ThinkingSnapshot:
    thinking_time: float
    hypotheses: int
    depth: int
    thinking_score: float
    thinking_flag: bool


class ThinkingMonitor:
    """Lightweight metacognitive tracker for (Reflect/Reason) activity."""

    def __init__(self, t_reflect_weight: float = 1.0, t_reason_weight: float = 1.2) -> None:
        self.t_reflect_weight = t_reflect_weight
        self.t_reason_weight = t_reason_weight
        self.begin_cycle()

    def begin_cycle(self) -> None:
        self._t_reflect = 0.0
        self._t_reason = 0.0
        self._t_reflect_start: float | None = None
        self._t_reason_start: float | None = None
        self._hypotheses = 0
        self._depth = 0

    # ---- hooks ----
    def on_reflect_start(self) -> None:
        if self._t_reflect_start is None:
            self._t_reflect_start = time.time()

    def on_reflect_end(self) -> None:
        if self._t_reflect_start is not None:
            self._t_reflect += time.time() - self._t_reflect_start
            self._t_reflect_start = None

    def on_reason_start(self) -> None:
        if self._t_reason_start is None:
            self._t_reason_start = time.time()

    def on_reason_end(self) -> None:
        if self._t_reason_start is not None:
            self._t_reason += time.time() - self._t_reason_start
            self._t_reason_start = None

    def on_hypothesis_tested(self, n: int = 1) -> None:
        self._hypotheses += max(0, int(n))

    def set_depth(self, n_steps_before_act: int) -> None:
        self._depth = max(0, int(n_steps_before_act))

    # ---- compute ----
    def snapshot(self) -> ThinkingSnapshot:
        # ensure to close any open spans (best-effort)
        self.on_reflect_end()
        self.on_reason_end()
        weighted_time = (
            self.t_reflect_weight * self._t_reflect
            + self.t_reason_weight * self._t_reason
        )
        time_term = min(1.0, weighted_time / 2.0)  # ~2s -> max time contribution
        hyp_term = min(1.0, self._hypotheses / 5.0)
        depth_term = min(1.0, self._depth / 4.0)
        score = max(0.0, min(1.0, 0.6 * time_term + 0.25 * hyp_term + 0.15 * depth_term))
        flag = (weighted_time > 0.25) or (self._hypotheses >= 1)
        return ThinkingSnapshot(
            thinking_time=weighted_time,
            hypotheses=self._hypotheses,
            depth=self._depth,
            thinking_score=score,
            thinking_flag=flag,
        )
