import os
import json
from typing import Dict, Any, List

class Consolidator:
    """Very small heuristic consolidator that extracts lessons from fresh memories."""

    def __init__(self, memory_store, state_path: str = "data/consolidator.json"):
        self.memory = memory_store
        self.path = state_path
        self.state: Dict[str, Any] = {"last_ts": 0.0, "lessons": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                self.state = {"last_ts": 0.0, "lessons": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

    def run_once_now(self) -> Dict[str, Any]:
        recents = [m for m in self.memory.get_recent_memories(200) if m.get("ts", 0) > self.state.get("last_ts", 0)]
        if not recents:
            return {"lessons": [], "processed": 0}

        lessons: List[str] = []
        reflective = [m for m in recents if (m.get("kind") or "").lower() in {"reflection", "lesson"}]
        for mem in reflective:
            text = (mem.get("text") or "").strip()
            if not text:
                continue
            lesson = text
            if len(text) > 160:
                lesson = text[:157] + "…"
            if lesson not in lessons:
                lessons.append(lesson)

        interactions = [m for m in recents if (m.get("kind") or "").lower() == "interaction"]
        if interactions and len(lessons) < 3:
            last_msg = interactions[-1].get("text", "")
            if last_msg:
                lessons.append(f"Noter l'importance de répondre à: {last_msg[:80]}")

        max_ts = max((m.get("ts", 0) for m in recents), default=self.state.get("last_ts", 0))
        self.state["last_ts"] = max(self.state.get("last_ts", 0), max_ts)
        if lessons:
            self.state.setdefault("lessons", []).extend(lessons)
            self.state["lessons"] = self.state["lessons"][-200:]
        self._save()
        return {"lessons": lessons, "processed": len(recents)}
