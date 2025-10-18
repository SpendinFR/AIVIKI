"""Semantic memory maintenance pipeline with concept, episodic and summarization tasks."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import random
import time

from .summarizer import ProgressiveSummarizer, SummarizerConfig


@dataclass
class TaskStats:
    """Lightweight container for reporting task executions."""

    last_run: float
    iterations: int = 0
    last_result: Optional[Any] = None


class SemanticMemoryManager:
    """Coordinate concept updates, episodic linking and summarization."""

    def __init__(
        self,
        *,
        memory_store: Any,
        concept_store: Optional[Any] = None,
        episodic_linker: Optional[Any] = None,
        consolidator: Optional[Any] = None,
        summarizer: Optional[ProgressiveSummarizer] = None,
        summarize_period_s: int = 60 * 30,
        concept_period_s: int = 10 * 60,
        episodic_period_s: int = 15 * 60,
        jitter_frac: float = 0.25,
        summarizer_config: Optional[SummarizerConfig] = None,
        llm_summarize_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self.m = memory_store
        self.c = concept_store
        self.e = episodic_linker
        self.consolidator = consolidator
        self.concept_period = max(0, concept_period_s)
        self.episodic_period = max(0, episodic_period_s)
        self.summarize_period = max(0, summarize_period_s)
        self.jitter_frac = max(0.0, min(1.0, jitter_frac))
        self.summarizer = summarizer or ProgressiveSummarizer(
            memory_store,
            concept_store=concept_store,
            config=summarizer_config,
            llm_summarize_fn=llm_summarize_fn,
        )

        now = self._now()
        self.next_concept = now
        self.next_episodic = now
        self.next_summarize = now

        self.concept_task = TaskStats(last_run=now)
        self.episodic_task = TaskStats(last_run=now)
        self.summary_task = TaskStats(last_run=now)

    # ------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Run maintenance tasks according to their schedule."""

        now = self._now(now)
        stats: Dict[str, Any] = {"ts": now}

        if self.concept_period and now >= self.next_concept:
            try:
                stats["concepts"] = self._run_concept_update(now)
            finally:
                self.next_concept = now + self._jitter(self.concept_period)
                self.concept_task.last_run = now
                self.concept_task.iterations += 1

        if self.episodic_period and now >= self.next_episodic:
            try:
                stats["episodic"] = self._run_episodic_linking(now)
            finally:
                self.next_episodic = now + self._jitter(self.episodic_period)
                self.episodic_task.last_run = now
                self.episodic_task.iterations += 1

        if self.summarize_period and now >= self.next_summarize:
            try:
                step_stats = self.summarizer.step(now)
                stats["summaries"] = step_stats
            finally:
                self.next_summarize = now + self._jitter(self.summarize_period)
                self.summary_task.last_run = now
                self.summary_task.iterations += 1
                self.summary_task.last_result = stats.get("summaries")

        return stats

    # ------------------------------------------------------------------
    def on_new_items(self, urgency: float = 0.5) -> None:
        """Nudge schedules so new items are processed sooner."""

        now = self._now()
        urgency = max(0.0, min(1.0, urgency))
        if self.concept_period:
            self.next_concept = min(self.next_concept, now + urgency * self.concept_period * 0.5)
        if self.episodic_period:
            self.next_episodic = min(self.next_episodic, now + urgency * self.episodic_period * 0.5)
        if self.summarize_period:
            self.next_summarize = min(self.next_summarize, now + urgency * self.summarize_period * 0.5)

    # ------------------------------------------------------------------
    def _run_concept_update(self, now: float) -> int:
        # Decay + update from recent items
        if hasattr(self.c, "decay_tick"):
            try:
                self.c.decay_tick(now)
            except Exception:
                pass

        # pull recent items since last window
        window = now - 3600  # last hour
        try:
            recent = list(self.m.list_items({"newer_than_ts": window, "limit": 500}) or [])
        except Exception:
            recent = []
        if hasattr(self.c, "update_from_items"):
            try:
                self.c.update_from_items(recent)
            except Exception:
                pass
        return len(recent)

    def _run_episodic_linking(self, now: float) -> int:
        if not self.e:
            return 0
        window = now - 4 * 3600  # last 4 hours
        try:
            recent = list(self.m.list_items({"newer_than_ts": window, "limit": 800}) or [])
        except Exception:
            recent = []
        try:
            created = self.e.link(recent) or []
        except Exception:
            created = []
        return len(created)

    def _jitter(self, period: int) -> float:
        if period <= 0:
            return 0.0
        j = self.jitter_frac
        spread = period * j
        return period + random.uniform(-spread, +spread)

    def _now(self, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        if hasattr(self.m, "now"):
            try:
                return float(self.m.now())
            except Exception:
                pass
        return time.time()


__all__ = ["SemanticMemoryManager", "SummarizerConfig", "ProgressiveSummarizer"]
