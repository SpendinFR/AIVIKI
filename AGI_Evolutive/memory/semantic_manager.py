import time
from typing import List, Tuple
from .concept_store import ConceptStore, Concept, Relation
from .concept_extractor import ConceptExtractor
from .episodic_linker import EpisodicLinker


class SemanticMemoryManager:
    """Coordonne l'extraction de concepts et le chaînage épisodique."""

    def __init__(self, memory_system, architecture=None):
        self.memory = memory_system
        self.architecture = architecture
        self.store = ConceptStore()
        self.extractor = ConceptExtractor(self.memory)
        self.extractor.store = self.store
        self.linker = EpisodicLinker(self.memory)
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
