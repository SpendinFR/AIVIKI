import os
import json
import time
import uuid
from typing import Any, Dict, List

class MemoryStore:
    """Simple persistent append-only memory buffer used by the orchestrator."""

    def __init__(self, path: str = "data/memory_store.json", max_items: int = 5000, flush_every: int = 10):
        self.path = path
        self.max_items = max_items
        self.flush_every = max(1, flush_every)
        self.state: Dict[str, Any] = {"memories": []}
        self._dirty = 0
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                self.state = {"memories": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)
        self._dirty = 0

    # ------------------------------------------------------------------
    def add_memory(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(entry)
        ts = data.get("ts", time.time())
        data.setdefault("ts", ts)
        data.setdefault("id", f"mem_{int(ts*1000)}_{uuid.uuid4().hex[:6]}")
        self.state.setdefault("memories", []).append(data)
        if len(self.state["memories"]) > self.max_items:
            self.state["memories"] = self.state["memories"][-self.max_items:]
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self._save()
        return data

    def get_recent_memories(self, n: int = 50) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return list(self.state.get("memories", [])[-n:])

    def all_memories(self) -> List[Dict[str, Any]]:
        return list(self.state.get("memories", []))

    def flush(self):
        if self._dirty:
            self._save()
