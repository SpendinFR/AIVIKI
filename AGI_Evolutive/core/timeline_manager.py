from __future__ import annotations

import time
from typing import Dict, List


class TimelineManager:
    """Belief snapshots & deltas per topic, persisted in memory."""

    def __init__(self, memory_store=None) -> None:
        self.memory = memory_store

    def snapshot(self, topic: str, beliefs: List[Dict]) -> str:
        event_id = f"belief_snapshot:{int(time.time() * 1000)}"
        event = {
            "kind": "belief_snapshot",
            "id": event_id,
            "topic": topic,
            "beliefs": beliefs,
            "ts": time.time(),
        }
        if self.memory is not None:
            self.memory.add(event)
        return event_id

    def delta(self, topic: str, b1: List[Dict], b2: List[Dict]) -> Dict:
        s1 = {b.get("stmt"): b for b in b1}
        s2 = {b.get("stmt"): b for b in b2}
        added = [s2[k] for k in s2.keys() - s1.keys()]
        removed = [s1[k] for k in s1.keys() - s2.keys()]
        updated = [s2[k] for k in s1.keys() & s2.keys() if s1[k].get("conf") != s2[k].get("conf")]
        conf_drift = sum(
            abs(s2.get(k, {}).get("conf", 0.0) - s1.get(k, {}).get("conf", 0.0))
            for k in s1.keys() & s2.keys()
        )
        return {
            "kind": "belief_delta",
            "topic": topic,
            "added": added,
            "removed": removed,
            "updated": updated,
            "confidence_drift": conf_drift,
            "ts": time.time(),
        }

    def project(self, topic: str, gaps: List[str]) -> List[Dict]:
        return [{"goal_kind": "LearnConcept", "topic": topic, "concept": gap} for gap in gaps]
