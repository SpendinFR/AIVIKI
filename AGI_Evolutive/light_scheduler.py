"""Lightweight, tick-driven scheduler used by the orchestrator layer."""

import time
from typing import Any, Callable, Dict

__all__ = ["LightScheduler"]


class LightScheduler:
    """Minimal scheduler that runs jobs when :meth:`tick` is called.

    This helper is deliberately synchronous and state-less between ticks.  It
    targets integration tests and tight control loops where the caller already
    drives the execution.  The background, thread-based scheduler remains
    available under :mod:`AGI_Evolutive.runtime.scheduler`.
    """

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def register_job(self, name: str, interval_sec: int, func: Callable[[], None]) -> None:
        self.jobs[name] = {"interval": max(5, int(interval_sec)), "func": func, "last": 0.0}

    def tick(self) -> None:
        now = time.time()
        for job in self.jobs.values():
            if now - job["last"] >= job["interval"]:
                try:
                    job["func"]()
                finally:
                    job["last"] = now
