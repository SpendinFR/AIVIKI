import json
from typing import List, Dict, Any, Optional, Tuple
from .encoders import TinyEncoder, cosine


class InMemoryIndex:
    """
    Index vectoriel simple (RAM) + métadonnées.
    - add_document(text, meta)
    - search_text(query, top_k)
    Persistance JSON optionnelle (sans dépendances).
    """

    def __init__(self, encoder: Optional[TinyEncoder] = None):
        self.encoder = encoder or TinyEncoder()
        self._docs: List[Dict[str, Any]] = []  # {"id": int, "text": str, "meta": {...}, "vec": [float]}
        self._next_id: int = 1

    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        vec = self.encoder.encode(text or "")
        doc_id = self._next_id
        self._next_id += 1
        self._docs.append({"id": doc_id, "text": text or "", "meta": meta or {}, "vec": vec})
        return doc_id

    def _search_vec(self, qvec: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        scores = []
        for d in self._docs:
            s = cosine(qvec, d["vec"])
            scores.append((d["id"], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: max(1, top_k)]

    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = self.encoder.encode(query or "")
        hits = self._search_vec(qvec, top_k=top_k)
        # Résolution en docs
        out = []
        by_id = {d["id"]: d for d in self._docs}
        for did, score in hits:
            d = by_id.get(did)
            if not d:
                continue
            out.append({
                "id": did,
                "score": float(round(score, 4)),
                "text": d["text"],
                "meta": dict(d["meta"]) if d["meta"] else {},
            })
        return out

    # ---------- Persistance (optionnelle) ----------
    def save(self, path: str):
        payload = {
            "next_id": self._next_id,
            "docs": [{"id": d["id"], "text": d["text"], "meta": d["meta"], "vec": d["vec"]} for d in self._docs],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._next_id = int(payload.get("next_id", 1))
            self._docs = []
            for d in payload.get("docs", []):
                # sécurité minimale
                if not isinstance(d, dict):
                    continue
                self._docs.append({
                    "id": int(d.get("id", 0)),
                    "text": str(d.get("text", "")),
                    "meta": dict(d.get("meta", {})),
                    "vec": list(d.get("vec", [])),
                })
        except FileNotFoundError:
            # premier run: normal
            pass
