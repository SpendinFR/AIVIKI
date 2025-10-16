try:
    import json
except ImportError as exc:  # pragma: no cover - standard library is expected
    raise RuntimeError("The json module is required for VectorStore operations") from exc

try:
    from math import sqrt
except ImportError as exc:  # pragma: no cover - standard library is expected
    raise RuntimeError("math.sqrt is required for VectorStore operations") from exc

import os
from typing import Dict, List, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize

from AGI_Evolutive.core.config import cfg

_DIR = cfg()["VECTOR_DIR"]


def _tokenize(text: str) -> Dict[str, int]:
    tokens: Dict[str, int] = {}
    for word in (text or "").lower().split():
        tokens[word] = tokens.get(word, 0) + 1
    return tokens


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    inter = set(a.keys()) & set(b.keys())
    dot = sum(a[key] * b[key] for key in inter)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Very small Bag-of-Words vector store."""

    def __init__(self) -> None:
        os.makedirs(_DIR, exist_ok=True)
        self.idx_path = os.path.join(_DIR, "index.json")
        self.index: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.idx_path):
            try:
                with open(self.idx_path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                    self.index = {doc: {term: int(freq) for term, freq in vec.items()} for doc, vec in raw.items()}
            except Exception:
                pass

    def _save(self) -> None:
        with open(self.idx_path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(self.index), handle, ensure_ascii=False, indent=2)

    def upsert(self, doc_id: str, text: str) -> None:
        self.index[doc_id] = _tokenize(text)
        self._save()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q_vec = _tokenize(query)
        scored = [(doc_id, _cosine(q_vec, vec)) for doc_id, vec in self.index.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
