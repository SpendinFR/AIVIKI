from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
        self._habit_strength_source: Optional[Any] = None

    def register(self, fn: Collector):
        self.collectors.append(fn)

    def set_habit_strength_source(self, source: Any) -> None:
        """Alimente la force d'habitude via EvolutionManager ou un callable."""

        self._habit_strength_source = source

    def _key(self, t: Trigger) -> str:
    def _payload_fingerprint(self, payload: Any) -> str:
        try:
            serialized = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = repr(payload)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _key(self, t: Trigger) -> Optional[str]:
        # build a stable key per (type, salient meta/payload)
        src = t.meta.get("source", "unknown")
        base_info: Any = t.meta.get("hash")
        if not base_info and isinstance(t.payload, dict):
            base_info = t.payload.get("id")
        if base_info:
            base = str(base_info)
        elif t.payload is not None:
            base = self._payload_fingerprint(t.payload)
        else:
            return None
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
        habit_key = m.get("habit_key") or m.get("key")
        if not habit_key and t.payload and isinstance(t.payload, dict):
            habit_key = t.payload.get("habit_key") or t.payload.get("id")
        strength = self._lookup_habit_strength(habit_key) if habit_key else 0.0
        if strength:
            m["habit_strength"] = strength
        # emotion influence is applied in scoring (not stored)
        return t

    def _lookup_habit_strength(self, key: Any) -> float:
        if key is None or self._habit_strength_source is None:
            return 0.0
        source = self._habit_strength_source
        try:
            if callable(source):
                return float(source(key))
            if isinstance(source, dict):
                return float(source.get(key, 0.0))
            getter = getattr(source, "get", None)
            if callable(getter):
                return float(getter(key, 0.0))
        except Exception:
            return 0.0
        return 0.0

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
                    if key is not None:
                        if self.cooldown_cache.get(key, 0) + 1.5 > now:
                            continue
                        self.cooldown_cache[key] = now
                    scored.append(ScoredTrigger(trigger=t, priority=pr))
            except Exception:
                continue
        # preemption rules
        scored.sort(key=lambda s: s.priority, reverse=True)
        return scored
