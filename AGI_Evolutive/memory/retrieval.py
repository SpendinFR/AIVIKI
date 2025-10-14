import time
from typing import Dict, Any, Optional, List
from .encoders import TinyEncoder
from .indexing import InMemoryIndex


class MemoryRetrieval:
    """
    Façade Retrieval:
    - add_interaction(user, agent, extra)
    - add_document(text, title, source)
    - search_text(query, top_k)
    """

    def __init__(self, encoder: Optional[TinyEncoder] = None, index: Optional[InMemoryIndex] = None):
        self.encoder = encoder or TinyEncoder()
        self.index = index or InMemoryIndex(self.encoder)

    # -------- Ajout ----------
    def add_interaction(self, user: str, agent: str, extra: Optional[Dict[str, Any]] = None) -> int:
        text = f"[USER] {user}\n[AGENT] {agent}"
        meta = {"type": "interaction", "ts": time.time()}
        if extra:
            meta.update(extra)
        return self.index.add_document(text, meta=meta)

    def add_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None) -> int:
        meta = {"type": "doc"}
        if title:
            meta["title"] = title
        if source:
            meta["source"] = source
        meta["ts"] = time.time()
        return self.index.add_document(text, meta=meta)

    # -------- Recherche ----------
    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.index.search_text(query, top_k=top_k)

    # -------- Persistance (optionnelle) ----------
    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)
