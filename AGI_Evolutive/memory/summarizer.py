"""
Progressive summarization and compaction pipeline for long-term memory.


Design goals
------------
1) Turn old raw memories into hierarchical digests: daily → weekly → monthly
2) Keep salient facts (concepts, preferences, decisions) while dropping noise
3) Maintain lineage for drill-down, and mark raw items as compressed
4) TTL policies by memory kind, with pin/keep overrides
5) Idempotent runs: safe to call repeatedly (e.g., on a cron or manager tick)


Integration contract (MemoryStore protocol)
-------------------------------------------
The summarizer expects a `memory_store` object exposing:


- list_items(filter: dict) -> Iterable[dict]
  Supported filters (all optional):
    kind: str | list[str]
    older_than_ts: float
    newer_than_ts: float
    not_compressed: bool  # only items where item.get("compressed_into") is falsy
    limit: int


- add_item(item: dict) -> str
  Returns new item id. The item must at least define: kind, ts, text.


- update_item(id: str, patch: dict) -> None


- get_item(id: str) -> dict


- now() -> float


The item schema is flexible but these fields are leveraged if present:
    id: str
    ts: float (epoch seconds)
    kind: str (e.g., "interaction", "thought", "episode", "digest.daily", ...)
    text: str
    concepts: list[str]
    salience: float (0..1)
    pinned: bool
    tags: list[str]
    metadata: dict (e.g., {"emotion": "happy", "context": "chat", ...})
    compressed_into: str | None
    lineage: list[str] (children item ids consolidated into this digest)
    expiry_ts: float | None (optional TTL)


The summarizer optionally uses a ConceptStore-like object:
    - get_concept_weight(concept: str) -> float  # salience/support (0..1+)
  If not provided, a neutral weight of 1.0 is assumed.


No external ML dependency: we implement deterministic summarization with a small
scoring heuristic. You can plug an LLM by passing `llm_summarize_fn` if desired.
"""
from __future__ import annotations


from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence
import time


DAY_SECONDS = 24 * 3600


# ----------------------
# Configuration defaults
# ----------------------
@dataclass
class SummarizerConfig:
    """Configuration for :class:`ProgressiveSummarizer`."""

    # Age thresholds (days)
    daily_after_days: int = 7  # raw → daily after 7 days
    weekly_after_days: int = 21  # aggregate dailies older than 21 days
    monthly_after_days: int = 90  # aggregate weeklies older than 90 days

    # Limits
    max_raw_per_day: int = 400  # cap for daily window
    max_daily_per_week: int = 14  # collect up to N daily digests per weekly
    max_weekly_per_month: int = 8  # collect up to N weekly digests per monthly

    # Minimum number of source items before emitting a digest
    min_items_daily: int = 1
    min_items_weekly: int = 1
    min_items_monthly: int = 1

    # TTL policies (days) – ``None`` disables TTL for that level
    ttl_daily_days: Optional[int] = 365
    ttl_weekly_days: Optional[int] = 730
    ttl_monthly_days: Optional[int] = None

    # Fallback summary formatting
    max_summary_bullets: int = 6
    max_bullet_chars: int = 280


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ProgressiveSummarizer:
    """Hierarchical summarization pipeline for long-term memories."""

    def __init__(
        self,
        memory_store: object,
        *,
        concept_store: Optional[object] = None,
        config: Optional[SummarizerConfig] = None,
        llm_summarize_fn: Optional[Callable[[str, Sequence[Mapping[str, object]]], str]] = None,
    ) -> None:
        self.m = memory_store
        self.c = concept_store
        self.config = config or SummarizerConfig()
        self.llm_summarize_fn = llm_summarize_fn
        self._known_fingerprints: set[str] = set()
        self._hydrate_fingerprint_cache()

    # ------------------------------------------------------------------
    def step(self, now: Optional[float] = None) -> MutableMapping[str, object]:
        """Run one maintenance pass of the summarization pipeline."""

        now = self._now(now)
        stats: MutableMapping[str, object] = {
            "ts": now,
            "daily": self._promote_raw_to_daily(now),
            "weekly": self._promote_daily_to_weekly(now),
            "monthly": self._promote_weekly_to_monthly(now),
        }
        return stats

    # ------------------------------------------------------------------
    def _hydrate_fingerprint_cache(self) -> None:
        levels = ("digest.daily", "digest.weekly", "digest.monthly")
        for kind in levels:
            try:
                existing = self.m.list_items({"kind": kind, "limit": 500})
            except Exception:
                existing = []
            for item in existing or []:
                fp = self._extract_fingerprint(item)
                if fp:
                    self._known_fingerprints.add(fp)

    # ------------------------------------------------------------------
    def _promote_raw_to_daily(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.daily_after_days * DAY_SECONDS
        try:
            raw_items = list(
                self.m.list_items(
                    {
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_raw_per_day,
                    }
                )
                or []
            )
        except Exception:
            raw_items = []
        candidates = [item for item in raw_items if self._is_raw_item(item)]
        buckets = self._bucket_by_period(candidates, period="daily")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_daily:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="daily",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(candidates),
            "created": created,
            "skipped": skipped,
        }

    def _promote_daily_to_weekly(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.weekly_after_days * DAY_SECONDS
        try:
            daily_items = list(
                self.m.list_items(
                    {
                        "kind": "digest.daily",
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_daily_per_week,
                    }
                )
                or []
            )
        except Exception:
            daily_items = []
        buckets = self._bucket_by_period(daily_items, period="weekly")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_weekly:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="weekly",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(daily_items),
            "created": created,
            "skipped": skipped,
        }

    def _promote_weekly_to_monthly(self, now: float) -> MutableMapping[str, object]:
        threshold = now - self.config.monthly_after_days * DAY_SECONDS
        try:
            weekly_items = list(
                self.m.list_items(
                    {
                        "kind": "digest.weekly",
                        "older_than_ts": threshold,
                        "not_compressed": True,
                        "limit": self.config.max_weekly_per_month,
                    }
                )
                or []
            )
        except Exception:
            weekly_items = []
        buckets = self._bucket_by_period(weekly_items, period="monthly")
        created = 0
        skipped = 0
        for (start_ts, end_ts), bucket in buckets.items():
            if len(bucket) < self.config.min_items_monthly:
                skipped += 1
                continue
            digest_id = self._materialize_digest(
                level="monthly",
                items=bucket,
                start_ts=start_ts,
                end_ts=end_ts,
                now=now,
            )
            if digest_id:
                created += 1
            else:
                skipped += 1
        return {
            "candidates": len(weekly_items),
            "created": created,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    def _materialize_digest(
        self,
        *,
        level: str,
        items: Sequence[Mapping[str, object]],
        start_ts: float,
        end_ts: float,
        now: float,
    ) -> Optional[str]:
        child_ids = sorted(
            str(item.get("id"))
            for item in items
            if isinstance(item, Mapping) and item.get("id")
        )
        if not child_ids:
            return None
        fingerprint = self._fingerprint(child_ids)
        if fingerprint in self._known_fingerprints:
            return None
        summary = self._compose_summary(level, items, start_ts, end_ts)
        if not summary.strip():
            return None
        digest_ts = min(end_ts - 1.0, now)
        metadata = {
            "digest_level": level,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "source_count": len(child_ids),
            "fingerprint": fingerprint,
        }
        payload = {
            "kind": f"digest.{level}",
            "ts": digest_ts,
            "text": summary,
            "concepts": self._merge_list_field(items, "concepts"),
            "tags": self._merge_list_field(items, "tags"),
            "salience": self._aggregate_salience(items),
            "lineage": child_ids,
            "metadata": metadata,
        }
        ttl_days = self._ttl_days_for(level)
        if ttl_days is not None:
            payload["expiry_ts"] = digest_ts + ttl_days * DAY_SECONDS
        try:
            digest_id = self.m.add_item(payload)
        except Exception:
            return None
        self._known_fingerprints.add(fingerprint)
        for item in items:
            item_id = item.get("id") if isinstance(item, Mapping) else None
            if not item_id:
                continue
            patch = {"compressed_into": digest_id}
            try:
                self.m.update_item(str(item_id), patch)
            except Exception:
                continue
        return str(digest_id)

    # ------------------------------------------------------------------
    def _compose_summary(
        self,
        level: str,
        items: Sequence[Mapping[str, object]],
        start_ts: float,
        end_ts: float,
    ) -> str:
        label = self._format_period(level, start_ts, end_ts)
        structured_items = [self._prepare_summary_item(item, level) for item in items]
        structured_items = [item for item in structured_items if item["text"]]
        if not structured_items:
            return ""
        if self.llm_summarize_fn is not None:
            try:
                return str(self.llm_summarize_fn(level, structured_items))
            except Exception:
                pass
        structured_items.sort(key=lambda payload: payload["score"], reverse=True)
        bullets = []
        limit = max(1, self.config.max_summary_bullets)
        for payload in structured_items[:limit]:
            text = payload["text"]
            if len(text) > self.config.max_bullet_chars:
                text = text[: self.config.max_bullet_chars - 1].rstrip() + "…"
            bullets.append(f"- {text}")
        concepts = self._merge_list_field(items, "concepts")
        concept_fragment = ""
        if concepts:
            concept_fragment = f"\nConcepts marquants: {', '.join(concepts[:8])}"
        return f"{label}\n" + "\n".join(bullets) + concept_fragment

    def _prepare_summary_item(self, item: Mapping[str, object], level: str) -> Mapping[str, object]:
        text = str(item.get("text", "")).strip()
        score = self._score_item(item)
        return {
            "id": item.get("id"),
            "text": text,
            "score": score,
            "level": level,
            "kind": item.get("kind"),
        }

    # ------------------------------------------------------------------
    def _bucket_by_period(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        period: str,
    ) -> MutableMapping[tuple[float, float], List[Mapping[str, object]]]:
        buckets: MutableMapping[tuple[float, float], List[Mapping[str, object]]] = defaultdict(list)
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ts = _safe_float(item.get("ts"), default=0.0)
            if ts <= 0.0:
                continue
            start_ts, end_ts = self._period_bounds(ts, period)
            buckets[(start_ts, end_ts)].append(item)
        return buckets

    def _period_bounds(self, ts: float, period: str) -> tuple[float, float]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if period == "daily":
            start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == "weekly":
            start = dt - timedelta(days=dt.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        elif period == "monthly":
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        else:
            raise ValueError(f"Unknown period: {period}")
        return (start.timestamp(), end.timestamp())

    def _format_period(self, level: str, start_ts: float, end_ts: float) -> str:
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        if level == "daily":
            return f"Synthèse du {start.strftime('%Y-%m-%d')}"
        if level == "weekly":
            iso_year, iso_week, _ = start.isocalendar()
            return f"Synthèse hebdomadaire {iso_year}-W{iso_week:02d}"
        if level == "monthly":
            return f"Synthèse mensuelle {start.strftime('%Y-%m')}"
        return f"Synthèse ({level}) {start.strftime('%Y-%m-%d')}"  # fallback

    # ------------------------------------------------------------------
    def _merge_list_field(self, items: Sequence[Mapping[str, object]], field: str) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for item in items:
            values = item.get(field) if isinstance(item, Mapping) else None
            if not isinstance(values, Iterable):
                continue
            for value in values:
                if not value:
                    continue
                key = str(value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(key)
        return merged

    def _aggregate_salience(self, items: Sequence[Mapping[str, object]]) -> float:
        if not items:
            return 0.0
        scores = [self._score_item(item) for item in items]
        return max(0.0, min(1.0, sum(scores) / max(1, len(scores))))

    def _score_item(self, item: Mapping[str, object]) -> float:
        base = _safe_float(item.get("salience"), default=0.3)
        concepts = item.get("concepts") if isinstance(item, Mapping) else None
        concept_weight = 1.0
        if concepts and isinstance(concepts, Iterable):
            weights = []
            for concept in concepts:
                weight = self._concept_weight(str(concept))
                if weight is not None:
                    weights.append(float(weight))
            if weights:
                concept_weight = sum(weights) / len(weights)
        ts = _safe_float(item.get("ts"), default=self._now())
        age_days = max(0.0, (self._now() - ts) / DAY_SECONDS)
        recency = math.exp(-age_days / 60.0)  # slow decay for archival content
        score = 0.6 * base + 0.3 * concept_weight + 0.1 * recency
        return max(0.0, min(1.5, score))

    def _concept_weight(self, concept: str) -> float:
        if not self.c:
            return 1.0
        if hasattr(self.c, "get_concept_weight"):
            try:
                weight = self.c.get_concept_weight(concept)  # type: ignore[attr-defined]
                return _safe_float(weight, default=1.0)
            except Exception:
                return 1.0
        concepts = getattr(self.c, "concepts", None)
        if isinstance(concepts, Mapping):
            payload = concepts.get(concept)
            if payload is None and hasattr(self.c, "_find_by_label"):
                try:
                    cid = self.c._find_by_label(concept)  # type: ignore[attr-defined]
                    if cid:
                        payload = concepts.get(cid)
                except Exception:
                    payload = None
            if payload is not None:
                if hasattr(payload, "salience"):
                    return _safe_float(getattr(payload, "salience"), default=1.0)
                if isinstance(payload, Mapping) and "salience" in payload:
                    return _safe_float(payload.get("salience"), default=1.0)
        return 1.0

    # ------------------------------------------------------------------
    def _is_raw_item(self, item: Mapping[str, object]) -> bool:
        if not isinstance(item, Mapping):
            return False
        if item.get("pinned"):
            return False
        if item.get("compressed_into"):
            return False
        kind = str(item.get("kind", ""))
        return not kind.startswith("digest.")

    def _extract_fingerprint(self, item: Mapping[str, object]) -> Optional[str]:
        if not isinstance(item, Mapping):
            return None
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            fp = metadata.get("fingerprint")
            if fp:
                return str(fp)
        lineage = item.get("lineage")
        if isinstance(lineage, Iterable):
            child_ids = [str(cid) for cid in lineage if cid]
            if child_ids:
                return self._fingerprint(sorted(child_ids))
        return None

    def _ttl_days_for(self, level: str) -> Optional[int]:
        if level == "daily":
            return self.config.ttl_daily_days
        if level == "weekly":
            return self.config.ttl_weekly_days
        if level == "monthly":
            return self.config.ttl_monthly_days
        return None

    def _fingerprint(self, texts: Sequence[str]) -> str:
        h = 0
        for text in texts:
            data = str(text).encode("utf-8", errors="ignore")
            digest = hashlib.blake2b(data, digest_size=8).digest()
            h ^= int.from_bytes(digest, "big", signed=False)
        return f"fp{h:016x}"

    def _now(self, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        if hasattr(self.m, "now"):
            try:
                return float(self.m.now())
            except Exception:
                pass
        return time.time()


__all__ = ["SummarizerConfig", "ProgressiveSummarizer"]
