import time
from collections import Counter
from typing import Any, Dict, List


class Consolidator:
    """Aggregate recent memories into lessons and improvement proposals."""

    def __init__(self, memory_store) -> None:
        self.memory = memory_store

    def run_once_now(self) -> Dict[str, Any]:
        recent = self.memory.get_recent_memories(n=100)
        topics: Counter[str] = Counter()
        errors = 0
        praises = 0
        for memo in recent:
            text = (memo.get("text") or "").lower()
            if "error" in memo.get("kind", ""):
                errors += 1
            if any(word in text for word in ["bravo", "bien", "good", "merci"]):
                praises += 1
            for word in text.split():
                if word.isalpha():
                    topics[word] += 1

        top_words = [word for word, _ in topics.most_common(5)]
        lessons: List[str] = []
        if errors >= 3:
            lessons.append("Pattern d’erreurs récurrentes détecté → ajouter étape de vérification.")
        if praises >= 3:
            lessons.append("Feedback positif récurrent → consolider la stratégie actuelle.")
        if top_words:
            lessons.append(f"Sujets dominants: {', '.join(top_words)}")

        proposals: List[Dict[str, Any]] = []
        if errors >= 3:
            proposals.append({"type": "update", "path": ["persona", "tone"], "value": "careful"})

        return {"lessons": lessons, "proposals": proposals, "ts": time.time()}
