from typing import List, Callable, Dict, Any
from dataclasses import dataclass
import time

from AGI_Evolutive.core.trigger_types import Trigger, TriggerType
from AGI_Evolutive.core.evaluation import unified_priority


Collector = Callable[[], List[Trigger]]


@dataclass
class ScoredTrigger:
    trigger: Trigger
    priority: float


class TriggerBus:
    """Collects triggers from various sources, normalizes meta, scores, arbitrates."""

    def __init__(self):
        self.collectors: List[Collector] = []
        self.cooldown_cache: Dict[str, float] = {}  # simple dedup/cooldown

    def register(self, fn: Collector):
        self.collectors.append(fn)

    def _key(self, t: Trigger) -> str:
        # build a stable key per (type, salient meta/payload)
        src = t.meta.get("source", "unknown")
        base = t.meta.get("hash") or (t.payload.get("id") if t.payload else "")
        return f"{src}:{t.type.name}:{base}:{t.meta.get('immediacy','')}:{t.meta.get('importance','')}"

    def _normalize(self, t: Trigger, valence: float = 0.0) -> Trigger:
        m = t.meta
        # defaults
        m.setdefault("importance", 0.5)
        m.setdefault("probability", 0.6)
        m.setdefault("reversibility", 1.0)
        m.setdefault("effort", 0.5)
        m.setdefault("uncertainty", 0.2)
        m.setdefault("immediacy", 0.2)
        m.setdefault("habit_strength", 0.0)
        m.setdefault("source", "system")
        # emotion influence is applied in scoring (not stored)
        return t

    def collect_and_score(self, valence: float = 0.0) -> List[ScoredTrigger]:
        now = time.time()
        scored: List[ScoredTrigger] = []
        for fn in self.collectors:
            try:
                for t in fn() or []:
                    t = self._normalize(t, valence=valence)
                    # hard overrides
                    if t.type is TriggerType.THREAT and t.meta.get("immediacy", 0.0) >= 0.8:
                        pr = 1.0
                    else:
                        pr = unified_priority(
                            impact=t.meta["importance"],
                            probability=t.meta["probability"],
                            reversibility=t.meta["reversibility"],
                            effort=t.meta["effort"],
                            uncertainty=t.meta["uncertainty"],
                            valence=valence,
                        )
                    key = self._key(t)
                    # cooldown 1.5s to avoid storms
                    if self.cooldown_cache.get(key, 0) + 1.5 > now:
                        continue
                    self.cooldown_cache[key] = now
                    scored.append(ScoredTrigger(trigger=t, priority=pr))
            except Exception:
                continue
        # preemption rules
        scored.sort(key=lambda s: s.priority, reverse=True)
        return scored
