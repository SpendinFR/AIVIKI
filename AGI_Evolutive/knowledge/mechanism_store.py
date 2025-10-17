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
