
from __future__ import annotations

"""
Unified MechanismStore for Mechanistic Actionable Insights (MAIs).

- Append-only JSONL event log for persistence (ops: add/update/retire/delete).
- Thread-safe cache with RLock.
- Backward compatible with legacy paths and ops.
- Rehydrates nested dataclasses (ImpactHypothesis, EvidenceRef) on load.
- Discovery helper: scan_applicable(..., include_status={'draft','active','ready'})

Environment variable override:
    MAI_STORE_PATH: absolute or relative path to the JSONL store.
Default path (if env var not set):
    data/runtime/mai_store.jsonl
If that file does not exist but legacy 'data/mai_store.jsonl' exists, it will use the legacy file.
"""

import getpass
import json
import os
import platform
import threading
import time
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional

# Import MAI and nested dataclasses. We only use types; persistence is via dicts.
from AGI_Evolutive.core.structures.mai import (
    EvidenceRef,
    ImpactHypothesis,
    MAI,
)

SCHEMA_VERSION = 1

# Default + legacy locations
_DEFAULT_PATH = Path(os.environ.get("MAI_STORE_PATH", ""))
if not _DEFAULT_PATH:
    _DEFAULT_PATH = Path("data/runtime/mai_store.jsonl")

_LEGACY_PATH = Path("data/mai_store.jsonl")


class MechanismStore:
    """Append-only JSONL store for :class:`MAI` objects (thread-safe)."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        audit_context: Optional[Mapping[str, object]] = None,
    ):
        # Resolve path with env/legacy fallback
        if path is not None:
            self.path = Path(path)
        else:
            # If default doesn't exist but legacy does, use legacy
            self.path = _DEFAULT_PATH
            if (not self.path.exists()) and _LEGACY_PATH.exists():
                self.path = _LEGACY_PATH

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._cache: Dict[str, MAI] = {}
        self._audit_context = self._build_audit_context(audit_context)
        self._hooks: Dict[str, List[Callable[[Mapping[str, object]], None]]] = defaultdict(list)

        if self.path.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Public API
    def add(self, mai: MAI) -> None:
        """Insert a new MAI into the store (or replace if same id)."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("add", mai)
            self._fire_hooks("add", {"id": mai.id, "mai": mai})

    def update(self, mai: MAI) -> None:
        """Update/replace an existing MAI by id."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("update", mai)
            self._fire_hooks("update", {"id": mai.id, "mai": mai})

    def retire(self, mai_id: str, reason: Optional[str] = None) -> None:
        """Soft-delete a MAI (kept in the event log, removed from cache)."""
        with self._lock:
            # Log whatever we know about it for audit (if present in cache, serialize full MAI; else id only)
            payload: Dict[str, object]
            if mai_id in self._cache:
                payload = asdict(self._cache[mai_id])
                # record retirement reason if provided
                if reason:
                    payload.setdefault("meta", {})
                    if isinstance(payload["meta"], dict):
                        payload["meta"]["retire_reason"] = reason
            else:
                payload = {"id": mai_id}
                if reason:
                    payload["retire_reason"] = reason

            self._append("retire", payload)
            self._cache.pop(mai_id, None)
            self._fire_hooks("retire", {"id": mai_id, "reason": reason})

    def delete(self, mai_id: str) -> None:
        """Hard-delete from the in-memory index. Logged in the JSONL as 'delete'."""
        with self._lock:
            self._cache.pop(mai_id, None)
            self._append("delete", {"id": mai_id})
            self._fire_hooks("delete", {"id": mai_id})

    def get(self, mai_id: str) -> Optional[MAI]:
        with self._lock:
            return self._cache.get(mai_id)

    def all(self) -> Iterator[MAI]:
        with self._lock:
            # Return an iterator over a snapshot list to avoid concurrent modification issues
            return iter(list(self._cache.values()))

    def scan_applicable(
        self,
        state: Mapping[str, object],
        predicate_registry: Mapping[str, object],
        *,
        include_status: Optional[Iterable[str]] = None,
    ) -> List[MAI]:
        """Return MAIs whose preconditions hold in the given state.

        Args:
            state: The current world-state (symbols/features) available to preconditions.
            predicate_registry: Functions/predicates available to evaluate preconditions.
            include_status: Optional set of allowed MAI.status values; defaults to {'draft','active','ready'}.
        """
        allowed_status = set(include_status or {"draft", "active", "ready"})
        winners: List[MAI] = []
        with self._lock:
            for mai in self._cache.values():
                if getattr(mai, "status", "active") not in allowed_status:
                    continue
                try:
                    if mai.is_applicable(state, predicate_registry):
                        winners.append(mai)
                except Exception:
                    # Defensive: skip malformed MAIs rather than crashing
                    continue
        return winners

    # ------------------------------------------------------------------
    # Extensibility / observability helpers
    def register_hook(self, event: str, callback: Callable[[Mapping[str, object]], None]) -> None:
        """Register a callback invoked with event payloads.

        Hooks receive a mapping containing at minimum the ``id`` of the MAI concerned.
        They execute within the store lock, so implementations should be fast/non-blocking.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._hooks[event].append(callback)

    def update_audit_context(self, extra: Mapping[str, object]) -> None:
        """Merge additional audit context that will be appended to future log records."""

        if not isinstance(extra, Mapping):
            raise TypeError("audit context must be a mapping")
        with self._lock:
            self._audit_context.update(extra)

    def annotate_metrics(
        self,
        mai_id: str,
        metrics: Mapping[str, float],
        *,
        freshness: Optional[float] = None,
        ttl: Optional[float] = None,
        extra: Optional[Mapping[str, object]] = None,
        history_limit: int = 50,
    ) -> None:
        """Annotate a MAI with external evaluation metrics without imposing aggregation logic."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        cleaned_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                cleaned_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        timestamp = time.time()
        with self._lock:
            mai = self._cache.get(mai_id)
            if not mai:
                raise KeyError(f"MAI '{mai_id}' not found")

            metadata = mai.metadata if isinstance(mai.metadata, dict) else {}
            metrics_block = metadata.setdefault("metrics", {})
            metrics_block["values"] = {**metrics_block.get("values", {}), **cleaned_metrics}
            metrics_block["updated_at"] = timestamp
            if freshness is not None:
                metrics_block["freshness"] = float(freshness)
            if ttl is not None:
                metrics_block["ttl"] = float(ttl)
            if extra and isinstance(extra, Mapping):
                extra_section = metrics_block.setdefault("extra", {})
                extra_section.update({str(k): v for k, v in extra.items()})

            history = metrics_block.setdefault("history", [])
            history.append(
                {
                    "time": timestamp,
                    "values": cleaned_metrics,
                    "freshness": freshness,
                    "ttl": ttl,
                }
            )
            if history_limit > 0 and len(history) > history_limit:
                del history[:-history_limit]

            mai.metadata = metadata
            mai.updated_at = timestamp
            self._cache[mai.id] = mai
            self._append("update", mai)
            self._fire_hooks(
                "annotate",
                {
                    "id": mai.id,
                    "metrics": cleaned_metrics,
                    "freshness": freshness,
                    "ttl": ttl,
                },
            )

    # ------------------------------------------------------------------
    # Persistence
    def _append(self, op: str, mai: Optional[MAI | Mapping[str, object]] = None) -> None:
        record: Dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "op": op,
            "time": time.time(),
        }
        audit_snapshot = dict(self._audit_context)
        audit_snapshot["pid"] = os.getpid()
        record["audit"] = audit_snapshot
        if mai is not None:
            if isinstance(mai, Mapping):
                record["mai"] = dict(mai)
            else:
                record["mai"] = asdict(mai)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_all(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                op = rec.get("op")
                payload = rec.get("mai") or {}
                if payload is None:
                    payload = {}

                if op == "delete" and isinstance(payload, dict) and payload.get("id") and not payload.get("status") and not payload.get("preconditions"):
                    # Minimal delete record {id: ...}
                    self._cache.pop(payload.get("id"), None)
                    continue

                # Accept both full MAI dict and minimal {'id': ...}
                mai: Optional[MAI] = None
                if isinstance(payload, dict) and payload.get("id"):
                    try:
                        payload = self._validate_payload(payload)
                        mai = self._rehydrate_mai(payload)
                    except Exception:
                        mai = None

                if op in {"add", "update"} and mai is not None:
                    self._cache[mai.id] = mai
                elif op in {"retire", "delete"}:
                    mai_id = payload.get("id") or (mai.id if mai else None)
                    if isinstance(mai_id, str):
                        self._cache.pop(mai_id, None)

    # ------------------------------------------------------------------
    # Maintenance
    def snapshot(self) -> None:
        """Rewrite the store to the current in-memory state (best-effort compaction)."""

        with self._lock:
            tmp_path = self.path.with_suffix(".snapshot")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for mai in self._cache.values():
                    record = {
                        "schema": SCHEMA_VERSION,
                        "op": "add",
                        "time": time.time(),
                        "audit": dict(self._audit_context),
                        "mai": asdict(mai),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path.replace(self.path)

    # ------------------------------------------------------------------
    # Rehydration helpers
    def _rehydrate_mai(self, payload: Mapping[str, object]) -> MAI:
        """Coerce a dict payload back into an MAI, rebuilding nested dataclasses."""
        data: Dict[str, object] = dict(payload)

        # Rehydrate expected_impact
        impact = data.get("expected_impact")
        if isinstance(impact, dict):
            data["expected_impact"] = ImpactHypothesis(**self._filter_fields(ImpactHypothesis, impact))

        # Rehydrate provenance_docs
        docs = data.get("provenance_docs") or []
        if isinstance(docs, list):
            data["provenance_docs"] = [
                d if isinstance(d, EvidenceRef) else EvidenceRef(**self._filter_fields(EvidenceRef, d))
                for d in docs
                if isinstance(d, (dict, EvidenceRef))
            ]

        # Filter to MAI dataclass fields to avoid constructor errors
        return MAI(**self._filter_fields(MAI, data))

    @staticmethod
    def _filter_fields(cls, payload: Mapping[str, object]) -> Dict[str, object]:
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in payload.items() if k in allowed}

    def _build_audit_context(
        self, audit_context: Optional[Mapping[str, object]]
    ) -> Dict[str, object]:
        default_context: Dict[str, object] = {
            "user": self._safe_getpass(),
            "host": platform.node() or "unknown-host",
            "runtime_version": os.environ.get("AGI_RUNTIME_VERSION", "unknown"),
        }
        if audit_context:
            default_context.update({str(k): v for k, v in audit_context.items()})
        return default_context

    @staticmethod
    def _safe_getpass() -> str:
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER", "unknown")

    def _fire_hooks(self, event: str, payload: Mapping[str, object]) -> None:
        callbacks = list(self._hooks.get(event, ()))
        if not callbacks:
            return
        for cb in callbacks:
            try:
                cb(payload)
            except Exception:
                continue

    def _validate_payload(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """Lightweight schema validation to guard against corrupted records."""

        data = dict(payload)

        if not isinstance(data.get("id"), str) or not data["id"].strip():
            raise ValueError("MAI payload missing valid 'id'")

        for key in ("tags", "provenance_docs", "provenance_episodes", "safety_invariants"):
            value = data.get(key)
            if value is None:
                data[key] = []
            elif not isinstance(value, list):
                data[key] = [value]

        metadata = data.get("metadata")
        if metadata is None:
            data["metadata"] = {}
        elif not isinstance(metadata, Mapping):
            data["metadata"] = {"_coerced_from": type(metadata).__name__}

        runtime_counters = data.get("runtime_counters")
        if runtime_counters is None or not isinstance(runtime_counters, Mapping):
            data["runtime_counters"] = {
                "activation": 0.0,
                "wins": 0.0,
                "benefit": 0.0,
                "regret": 0.0,
                "rollbacks": 0.0,
            }

        for ts_field in ("created_at", "updated_at"):
            value = data.get(ts_field)
            if value is not None:
                try:
                    data[ts_field] = float(value)
                except (TypeError, ValueError):
                    data.pop(ts_field, None)

        return data


__all__ = ["MAI", "ImpactHypothesis", "EvidenceRef", "MechanismStore"]
