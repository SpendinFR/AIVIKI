import time
from typing import Dict, Any, Optional, List, Tuple

from .encoders import TinyEncoder
from .indexing import InMemoryIndex
from .vector_store import VectorStore


class MemoryRetrieval:
    """
    Façade Retrieval:
    - add_interaction(user, agent, extra)
    - add_document(text, title, source)
    - search_text(query, top_k)
    """

    def __init__(
        self,
        encoder: Optional[TinyEncoder] = None,
        index: Optional[InMemoryIndex] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.encoder = encoder or TinyEncoder()
        self.index = index or InMemoryIndex(self.encoder)
        self.vector_store = vector_store

    # -------- Ajout ----------
    def add_interaction(self, user: str, agent: str, extra: Optional[Dict[str, Any]] = None) -> int:
        text = f"[USER] {user}\n[AGENT] {agent}"
        meta = {"type": "interaction", "ts": time.time()}
        if extra:
            meta.update(extra)
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"interaction::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    def add_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None) -> int:
        meta = {"type": "doc"}
        if title:
            meta["title"] = title
        if source:
            meta["source"] = source
        meta["ts"] = time.time()
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"doc::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    # -------- Recherche ----------
    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return self.index.search_text(query, top_k=top_k)

        try:
            hits = self.vector_store.search(query or "", k=top_k)
        except Exception:
            return self.index.search_text(query, top_k=top_k)

        results: List[Dict[str, Any]] = []
        for doc_id, score in hits:
            kind, doc_record = self._resolve_hit(doc_id)
            if not doc_record:
                continue
            payload = {
                "id": doc_record["id"],
                "score": float(round(score, 4)),
                "text": doc_record.get("text", ""),
                "meta": dict(doc_record.get("meta", {})),
            }
            if kind:
                payload.setdefault("meta", {})["vector_kind"] = kind
            results.append(payload)
        if not results:
            return self.index.search_text(query, top_k=top_k)
        return results

    # -------- Persistance (optionnelle) ----------
    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)

    # -------- Helpers ----------
    def _resolve_hit(self, identifier: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not identifier:
            return None, None
        raw = identifier
        kind: Optional[str] = None
        if "::" in raw:
            kind, raw = raw.split("::", 1)
        try:
            doc_id = int(raw)
        except ValueError:
            return kind, None
        record = self.index.get_document(doc_id)
        return kind, record
