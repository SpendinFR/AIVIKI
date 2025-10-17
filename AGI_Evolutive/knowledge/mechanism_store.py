"""Persistence layer for Mechanistic Actionable Insights (MAIs).

Entries are stored as JSON lines with a simple change-log format so that the
store can be replayed at start-up.  The JSON payloads are produced via
``dataclasses.asdict`` which flattens nested dataclasses.  When reading the log
back we therefore need to rehydrate those nested structures to recover the
original dataclass instances.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

from AGI_Evolutive.core.structures.mai import (
    EvidenceRef,
    ImpactHypothesis,
    MAI,
)

class MechanismStore:
    """Append-only JSONL store for :class:`MAI` objects."""

    DEFAULT_PATH = Path("data/runtime/mai_store.jsonl")

    def __init__(self, path: Path | str | None = None):
        resolved = Path(path) if path is not None else self.DEFAULT_PATH
        self.path = resolved
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, MAI] = {}
        if self.path.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Discovery helpers
    def scan_applicable(
        self,
        state: Mapping[str, object],
        predicate_registry: Mapping[str, object],
        *,
        include_status: Optional[Iterable[str]] = None,
    ) -> List[MAI]:
        allowed_status = set(include_status or {"draft", "active", "ready"})
        applicable: List[MAI] = []
        for mai in self._cache.values():
            if mai.status not in allowed_status:
                continue
            try:
                if mai.is_applicable(state, predicate_registry):
                    applicable.append(mai)
            except Exception:
                continue
        return applicable

    # ------------------------------------------------------------------
    # Persistence helpers
    def _append(self, op: str, mai: Optional[MAI | Mapping[str, object]] = None) -> None:
        record: Dict[str, object] = {"op": op}
        if mai is not None:
            if isinstance(mai, Mapping):
                record["mai"] = dict(mai)
            else:
                record["mai"] = asdict(mai)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rehydrate_mai(self, payload: Mapping[str, object]) -> MAI:
        data: Dict[str, object] = dict(payload)
        impact = data.get("expected_impact")
        if isinstance(impact, dict):
            data["expected_impact"] = ImpactHypothesis(**self._filter_fields(ImpactHypothesis, impact))
        docs = data.get("provenance_docs") or []
        if isinstance(docs, list):
            data["provenance_docs"] = [
                doc
                if isinstance(doc, EvidenceRef)
                else EvidenceRef(**self._filter_fields(EvidenceRef, doc))
                for doc in docs
            ]
        return MAI(**self._filter_fields(MAI, data))

    @staticmethod
    def _filter_fields(cls, payload: Mapping[str, object]) -> Dict[str, object]:
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in payload.items() if k in allowed}

    def _load_all(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                op = record.get("op")
                payload = record.get("mai")
                if not payload:
                    continue
                mai = self._rehydrate_mai(payload)
                if op in {"add", "update"}:
                    self._cache[mai.id] = mai
                elif op == "delete":
                    self._cache.pop(mai.id, None)

    # ------------------------------------------------------------------
    # CRUD operations
    def add(self, mai: MAI) -> None:
        self._cache[mai.id] = mai
        self._append("add", mai)

    def update(self, mai: MAI) -> None:
        self._cache[mai.id] = mai
        self._append("update", mai)

    def delete(self, mai_id: str) -> None:
        if mai_id in self._cache:
            del self._cache[mai_id]
        self._append("delete", {"id": mai_id})

    def get(self, mai_id: str) -> Optional[MAI]:
        return self._cache.get(mai_id)

    def all(self) -> Iterator[MAI]:
        return iter(self._cache.values())


__all__ = [
    "EvidenceRef",
    "ImpactHypothesis",
    "MAI",
    "MechanismStore",
]
