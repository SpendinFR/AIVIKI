import json
import os
import time
from collections import Counter
from typing import Any, Dict, List

from AGI_Evolutive.utils.jsonsafe import json_sanitize

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
            json.dump(json_sanitize(self.state), fh, ensure_ascii=False, indent=2)

    def run_once_now(self) -> Dict[str, Any]:
        recents = [
            m
            for m in self.memory.get_recent_memories(200)
            if m.get("ts", 0) > self.state.get("last_ts", 0)
        ]
        if not recents:
            return {"lessons": [], "processed": 0, "proposals": []}

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

        topics: Counter[str] = Counter()
        errors = 0
        praises = 0
        for mem in recents:
            text = (mem.get("text") or mem.get("content") or "").lower()
            kind = (mem.get("kind") or "").lower()
            if "error" in kind:
                errors += 1
            if any(word in text for word in ["bravo", "bien", "good", "merci"]):
                praises += 1
            for word in text.split():
                if word.isalpha() and len(word) >= 4:
                    topics[word] += 1

        if topics and len(lessons) < 5:
            dominant = ", ".join(word for word, _ in topics.most_common(5))
            lessons.append(f"Sujets dominants: {dominant}")

        proposals: List[Dict[str, Any]] = []
        if errors >= 3:
            proposals.append(
                {
                    "type": "update",
                    "path": ["persona", "tone"],
                    "value": "careful",
                    "reason": "Pattern d'erreurs récurrentes détecté",
                }
            )
        if praises >= 3:
            proposals.append(
                {
                    "type": "reinforce",
                    "path": ["strategies", "positive_feedback"],
                    "value": True,
                    "reason": "Feedback positif récurrent",
                }
            )

        max_ts = max((m.get("ts", 0) for m in recents), default=self.state.get("last_ts", 0))
        self.state["last_ts"] = max(self.state.get("last_ts", 0), max_ts)
        if lessons:
            self.state.setdefault("lessons", []).extend(lessons)
            self.state["lessons"] = self.state["lessons"][-200:]
        self._save()
        return {"lessons": lessons, "processed": len(recents), "proposals": proposals}
