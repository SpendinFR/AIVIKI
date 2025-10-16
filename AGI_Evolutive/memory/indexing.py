import json
from typing import List, Dict, Any, Optional, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize

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
        self._docs: List[Dict[str, Any]] = []  # {"id": int, "text": str, "meta": {"source": str}, "vec": List[float]}
        self._next_id: int = 1
        self._by_id: Dict[int, Dict[str, Any]] = {}

    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        vec = self.encoder.encode(text or "")
        doc_id = self._next_id
        self._next_id += 1
        record = {"id": doc_id, "text": text or "", "meta": meta or {}, "vec": vec}
        self._docs.append(record)
        self._by_id[doc_id] = record
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
        for did, score in hits:
            d = self._by_id.get(did)
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
            json.dump(json_sanitize(payload), f, ensure_ascii=False)

    def load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._next_id = int(payload.get("next_id", 1))
            self._docs = []
            self._by_id = {}
            for d in payload.get("docs", []):
                # sécurité minimale
                if not isinstance(d, dict):
                    continue
                record = {
                    "id": int(d.get("id", 0)),
                    "text": str(d.get("text", "")),
                    "meta": dict(d.get("meta", {})),
                    "vec": list(d.get("vec", [])),
                }
                self._docs.append(record)
                self._by_id[record["id"]] = record
        except FileNotFoundError:
            # premier run: normal
            pass

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self._by_id.get(int(doc_id))
