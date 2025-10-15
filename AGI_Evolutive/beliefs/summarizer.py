"""Batch summarizer producing hierarchical belief summaries."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from AGI_Evolutive.utils.jsonsafe import json_sanitize

from .graph import Belief, BeliefGraph


@dataclass
class SummaryConfig:
    timeframe: str  # "daily" | "weekly"
    window_seconds: float
    output_path: str


DEFAULT_CONFIGS = {
    "daily": SummaryConfig(timeframe="daily", window_seconds=60 * 60 * 24, output_path="data/summaries/daily.jsonl"),
    "weekly": SummaryConfig(
        timeframe="weekly",
        window_seconds=60 * 60 * 24 * 7,
        output_path="data/summaries/weekly.jsonl",
    ),
}


class BeliefSummarizer:
    """Aggregates beliefs into anchors and episodic summaries."""

    def __init__(self, graph: BeliefGraph) -> None:
        self.graph = graph

    def _select_beliefs(self, beliefs: Iterable[Belief], *, now: Optional[float] = None, window: float = 0.0) -> List[Belief]:
        now = now or time.time()
        selected: List[Belief] = []
        for belief in beliefs:
            if belief.stability == "anchor":
                selected.append(belief)
                continue
            if belief.updated_at >= now - window:
                selected.append(belief)
        return sorted(selected, key=lambda b: (b.stability, -b.confidence, -b.updated_at))

    def build_summary(self, timeframe: str, *, now: Optional[float] = None) -> Dict[str, List[Dict[str, object]]]:
        cfg = DEFAULT_CONFIGS.get(timeframe)
        if not cfg:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        now = now or time.time()
        beliefs = self.graph.all()
        selected = self._select_beliefs(beliefs, now=now, window=cfg.window_seconds)
        anchors: List[Dict[str, object]] = []
        episodes: List[Dict[str, object]] = []
        for belief in selected:
            bundle = {
                "subject": belief.subject,
                "relation": belief.relation,
                "value": belief.value,
                "confidence": round(belief.confidence, 3),
                "polarity": belief.polarity,
                "evidence": [e.source for e in belief.justifications],
                "updated_at": belief.updated_at,
                "temporal_segments": [seg.to_dict() for seg in belief.temporal_segments],
            }
            if belief.stability == "anchor":
                anchors.append(bundle)
            else:
                episodes.append(bundle)
        return {"anchors": anchors[:20], "episodes": episodes[:50]}

    def write_summary(self, timeframe: str, *, now: Optional[float] = None) -> Dict[str, List[Dict[str, object]]]:
        cfg = DEFAULT_CONFIGS[timeframe]
        summary = self.build_summary(timeframe, now=now)
        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        with open(cfg.output_path, "a", encoding="utf-8") as fh:
            row = {"time": now or time.time(), "summary": summary}
            fh.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")
        return summary


def run_batch(graph: BeliefGraph, timeframes: Iterable[str] = ("daily", "weekly")) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    summarizer = BeliefSummarizer(graph)
    now = time.time()
    return {tf: summarizer.write_summary(tf, now=now) for tf in timeframes}
