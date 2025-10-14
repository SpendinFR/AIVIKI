import os, json, time
from typing import Dict, Any, List
from collections import Counter

class ConceptExtractor:
    """
    Extrait des "concepts" (mots/phrases clés dominants) des souvenirs récents.
    Persiste dans data/concepts.json
    """
    def __init__(self, memory_store, data_path: str = "data/concepts.json"):
        self.memory = memory_store
        self.path = data_path
        self.state = {"concepts": {}, "updated_at": 0.0}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def extract_from_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        recents = self.memory.get_recent_memories(n=n)
        cnt = Counter()
        for m in recents:
            text = (m.get("text") or "").lower()
            for w in text.split():
                if w.isalpha() and len(w) >= 4:
                    cnt[w] += 1
        top = cnt.most_common(20)
        out = []
        for w,fq in top:
            score = min(1.0, 0.2 + fq/20.0)
            cur = self.state["concepts"].get(w, {"count": 0, "score": 0.0})
            cur["count"] += fq
            cur["score"] = max(cur["score"], score)
            self.state["concepts"][w] = cur
            out.append({"concept": w, "score": score})
        self.state["updated_at"] = time.time()
        self._save()
        return out
