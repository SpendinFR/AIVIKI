"""Persistence layer for Mechanistic Actionable Insights (MAIs).

This module provides lightweight dataclasses representing MAIs and a
`MechanismStore` class responsible for persisting them to disk.  Entries
are stored as JSON lines with a simple change-log format so that the store
can be replayed at start-up.  The JSON payloads are produced via
``dataclasses.asdict`` which flattens nested dataclasses.  When reading the
log back we therefore need to rehydrate those nested structures to recover
the original dataclass instances.

The ``MechanismStore`` implements that rehydration step in
:meth:`MechanismStore._load_all` by converting dictionaries into
``ImpactHypothesis`` and ``EvidenceRef`` objects before instantiating
``MAI``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional


@dataclass
class EvidenceRef:
    """Reference to an external document supporting an MAI."""

    source: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    kind: Optional[str] = None


@dataclass
class ImpactHypothesis:
    """Hypothesis describing the impact expected from applying an MAI."""

    trust_delta: float = 0.0
    confidence: float = 0.0
    rationale: Optional[str] = None
    caveats: Optional[str] = None


@dataclass
class MAI:
    """Mechanistic Actionable Insight representation."""

    id: str
    title: str = ""
    summary: str = ""
    status: str = "draft"
    expected_impact: ImpactHypothesis = field(default_factory=ImpactHypothesis)
    provenance_docs: List[EvidenceRef] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    owner: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


class MechanismStore:
    """Append-only JSONL store for :class:`MAI` objects."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, MAI] = {}
        if self.path.exists():
            self._load_all()

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
