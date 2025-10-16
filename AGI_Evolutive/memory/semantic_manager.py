import time
from typing import Any, Dict, List, Optional, Tuple

from .concept_store import ConceptStore, Concept, Relation
from .concept_extractor import ConceptExtractor
from .episodic_linker import EpisodicLinker
from .vector_store import VectorStore


class SemanticMemoryManager:
    """Coordonne l'extraction de concepts et le chaînage épisodique."""

    def __init__(
        self,
        memory_system,
        architecture=None,
        *,
        index_backend: Optional[VectorStore] = None,
    ):
        self.memory = memory_system
        self.architecture = architecture
        self.store = ConceptStore()
        self.extractor = ConceptExtractor(self.memory)
        self.extractor.store = self.store
        self.linker = EpisodicLinker(self.memory)
        self.index_backend = index_backend or VectorStore()
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}
        self.last_concept_step = 0.0
        self.last_link_step = 0.0
        self.concept_period = 5.0
        self.episodic_period = 5.0

    def step(self) -> None:
        now = time.time()
        try:
            if now - self.last_concept_step > self.concept_period:
                self.extractor.step(self.memory, max_batch=300)
                self.last_concept_step = now
        except Exception as exc:
            print(f"[semantic] concept step error: {exc}")
        try:
            if now - self.last_link_step > self.episodic_period:
                self.linker.step(self.memory, max_batch=200)
                self.last_link_step = now
        except Exception as exc:
            print(f"[semantic] episodic step error: {exc}")

    def get_top_concepts(self, k: int = 20) -> List[Concept]:
        return self.store.get_top_concepts(k)

    def neighbors(self, concept_label_prefix: str, k: int = 10) -> List[Tuple[Relation, Concept]]:
        candidates = self.store.find_by_label_prefix(concept_label_prefix, k=1)
        if not candidates:
            return []
        return self.store.neighbors(candidates[0].id, rtype=None, k=k)

    # ------------------------------------------------------------------
    # Vector index helpers
    def index_document(
        self,
        doc_id: str,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist ``text`` in the vector store and keep lightweight metadata."""

        if not doc_id:
            return
        try:
            self.index_backend.upsert(doc_id, text or "")
            if metadata:
                self._doc_metadata[doc_id] = dict(metadata)
        except Exception as exc:
            print(f"[semantic] vector index failure for {doc_id}: {exc}")

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search indexed documents using the vector backend."""

        try:
            hits = self.index_backend.search(query or "", k=k)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for doc_id, score in hits:
            payload: Dict[str, Any] = {
                "id": doc_id,
                "score": float(score),
            }
            if doc_id in self._doc_metadata:
                payload["metadata"] = dict(self._doc_metadata[doc_id])
            results.append(payload)
        return results
