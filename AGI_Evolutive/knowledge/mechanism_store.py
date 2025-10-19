
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

import json
import os
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

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

    def __init__(self, path: Path | str | None = None):
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

        if self.path.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Public API
    def add(self, mai: MAI) -> None:
        """Insert a new MAI into the store (or replace if same id)."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("add", mai)

    def update(self, mai: MAI) -> None:
        """Update/replace an existing MAI by id."""
        with self._lock:
            self._cache[mai.id] = mai
            self._append("update", mai)

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

    def delete(self, mai_id: str) -> None:
        """Hard-delete from the in-memory index. Logged in the JSONL as 'delete'."""
        with self._lock:
            self._cache.pop(mai_id, None)
            self._append("delete", {"id": mai_id})

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
    # Persistence
    def _append(self, op: str, mai: Optional[MAI | Mapping[str, object]] = None) -> None:
        record: Dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "op": op,
            "time": time.time(),
        }
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


__all__ = ["MAI", "ImpactHypothesis", "EvidenceRef", "MechanismStore"]
