from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional
import json, os, threading, time, pathlib

from AGI_Evolutive.core.structures.mai import MAI

_DEFAULT_PATH = os.environ.get("MAI_STORE_PATH", "data/mai_store.jsonl")

class MechanismStore:
    """
    Registre versionné des MAI.
    Format: JSONL (append-only). Chaque enregistrement: {"op":"add|update|retire","time":...,"mai":{...}}
    """
    def __init__(self, path: Optional[str]=None):
        self.path = path or _DEFAULT_PATH
        pathlib.Path(os.path.dirname(self.path) or ".").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, MAI] = {}
        self._load_all()

    # ------------- Persistence -------------
    def _load_all(self) -> None:
        self._cache.clear()
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                op = rec.get("op")
                d = rec.get("mai")
                if not d: continue
                m = MAI(**d)
                if op == "add" or op == "update":
                    self._cache[m.id] = m
                elif op == "retire" and m.id in self._cache:
                    del self._cache[m.id]

    def _append(self, op: str, mai: MAI) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            json.dump({"op": op, "time": time.time(), "mai": asdict(mai)}, f, ensure_ascii=False)
            f.write("\n")

    # ------------- API -------------
    def add(self, mai: MAI) -> None:
        with self._lock:
            self._cache[mai.id] = mai
            self._append("add", mai)

    def update(self, mai: MAI) -> None:
        with self._lock:
            if mai.id not in self._cache:
                self._cache[mai.id] = mai
            else:
                self._cache[mai.id] = mai
            self._append("update", mai)

    def retire(self, mai_id: str) -> None:
        with self._lock:
            if mai_id in self._cache:
                m = self._cache[mai_id]
                self._append("retire", m)
                del self._cache[mai_id]

    def all(self) -> List[MAI]:
        with self._lock:
            return list(self._cache.values())

    def scan_applicable(self, state: Dict[str, Any], predicate_registry: Dict[str, Any]) -> List[MAI]:
        winners: List[MAI] = []
        with self._lock:
            for m in self._cache.values():
                try:
                    if m.is_applicable(state, predicate_registry):
                        winners.append(m)
                except Exception:
                    continue
        return winners
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
