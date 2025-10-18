"""Scoring de saillance pour les items mémoire."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

try:  # Modules optionnels (config) peuvent manquer selon l'environnement
    from config import memory_flags as _mem_flags
except Exception:  # pragma: no cover - robustesse import
    _mem_flags = None  # type: ignore


def _norm01(value: float) -> float:
    """Normalise ``value`` dans [0, 1]."""

    if math.isnan(value):  # type: ignore[arg-type]
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class SalienceScorer:
    """Combine plusieurs signaux (récence, affect, etc.) en score 0..1."""

    def __init__(
        self,
        *,
        now: Optional[Callable[[], float]] = None,
        reward: Optional[Any] = None,
        goals: Optional[Any] = None,
        prefs: Optional[Any] = None,
    ) -> None:
        self.now: Callable[[], float] = now or time.time
        self.reward = reward
        self.goals = goals
        self.prefs = prefs

        self._weights = getattr(_mem_flags, "SALIENCE_WEIGHTS", {
            "recency": 0.25,
            "affect": 0.20,
            "reward": 0.15,
            "goal_rel": 0.15,
            "prefs": 0.15,
            "novelty": 0.07,
            "usage": 0.03,
        })
        self._half_lives = getattr(_mem_flags, "HALF_LIVES", {
            "default": 3 * 24 * 3600,
            "interaction": 2 * 24 * 3600,
            "episode": 7 * 24 * 3600,
            "digest.daily": 14 * 24 * 3600,
            "digest.weekly": 30 * 24 * 3600,
            "digest.monthly": 90 * 24 * 3600,
        })

    # ------------------------------------------------------------------
    def score(self, item: Dict[str, Any]) -> float:
        """Retourne la saillance globale d'un item mémoire."""

        if not item:
            return 0.0

        parts = {
            "recency": self._recency(item),
            "affect": self._affect(item),
            "reward": self._reward(item),
            "goal_rel": self._goal_rel(item),
            "prefs": self._prefs(item),
            "novelty": self._novelty(item),
            "usage": self._usage(item),
        }
        total_weight = sum(self._weights.values()) or 1.0
        score = 0.0
        for key, weight in self._weights.items():
            score += weight * parts.get(key, 0.0)
        return _norm01(score / total_weight)

    # ------------------------------------------------------------------
    def _recency(self, item: Dict[str, Any]) -> float:
        ts = self._timestamp(item)
        if ts is None:
            return 0.5
        age = max(0.0, self.now() - ts)
        label = str(item.get("kind") or item.get("type") or "default")
        half = float(self._half_lives.get(label, self._half_lives.get("default", 1.0)))
        if half <= 0.0:
            return 1.0
        return _norm01(math.pow(0.5, age / half))

    def _affect(self, item: Dict[str, Any]) -> float:
        affect = item.get("affect") or {}
        if isinstance(affect, dict):
            val = float(affect.get("valence", affect.get("val", 0.0)))
            aro = float(affect.get("arousal", affect.get("aro", 0.0)))
        elif isinstance(affect, (int, float)):
            val = float(affect)
            aro = 0.0
        else:
            val = 0.0
            aro = 0.0
        # map [-1,1]x[0,1] -> [0,1]
        base = (val + 1.0) / 2.0
        return _norm01(0.7 * base + 0.3 * max(0.0, aro))

    def _reward(self, item: Dict[str, Any]) -> float:
        # chercher un champ direct sinon interroger reward_engine
        if isinstance(item.get("reward"), (int, float)):
            r = float(item["reward"])  # attendu -1..+1
            return _norm01((r + 1.0) / 2.0)
        if self.reward and hasattr(self.reward, "recent_for"):
            try:
                r = float(self.reward.recent_for(item))  # duck-typed
                return _norm01((r + 1.0) / 2.0)
            except Exception:
                pass
        return 0.5  # neutre

    def _goal_rel(self, item: Dict[str, Any]) -> float:
        if self.goals and hasattr(self.goals, "relevance"):
            try:
                return _norm01(float(self.goals.relevance(item)))
            except Exception:
                return 0.0
        # heuristique: tags/metadata goal:true
        if item.get("metadata", {}).get("goal_related"):
            return 0.8
        return 0.0

    def _prefs(self, item: Dict[str, Any]) -> float:
        if not self.prefs:
            return 0.0
        try:
            concepts = item.get("concepts", []) or []
            tags = item.get("tags", []) or []
            return _norm01(float(self.prefs.get_affinity(concepts, tags)))
        except Exception:
            return 0.0

    def _novelty(self, item: Dict[str, Any]) -> float:
        # Par défaut: 0.5 (ni nouveau ni redondant). À améliorer (simhash/embeddings) si dispo.
        return float(item.get("novelty", 0.5)) if isinstance(item.get("novelty"), (int, float)) else 0.5

    def _usage(self, item: Dict[str, Any]) -> float:
        acc = float(item.get("access_count", 0))
        # boost saturé à 1.0 vers 10 accès
        return max(0.0, min(1.0, acc / 10.0))

    # ------------------------------------------------------------------
    def _timestamp(self, item: Dict[str, Any]) -> Optional[float]:
        ts = item.get("ts") or item.get("timestamp") or item.get("created_at")
        if isinstance(ts, (int, float)):
            return float(ts)
        return None
