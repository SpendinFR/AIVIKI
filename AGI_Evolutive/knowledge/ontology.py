"""Knowledge-layer helpers wrapping the belief ontology and entity linker."""

from __future__ import annotations

from typing import Any, Dict, Optional

from AGI_Evolutive.beliefs.entity_linker import EntityLinker as _BeliefEntityLinker
from AGI_Evolutive.beliefs.graph import BeliefGraph
from AGI_Evolutive.beliefs.ontology import Ontology as _BeliefOntology


class Ontology(_BeliefOntology):
    """Thin wrapper that defaults to the richer belief ontology."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if not args and not kwargs:
            default = _BeliefOntology.default()
            self.entity_types = dict(default.entity_types)
            self.relation_types = dict(default.relation_types)
            self.event_types = dict(default.event_types)


class EntityLinker(_BeliefEntityLinker):
    """Entity linker aware of the ontology and the belief graph."""

    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        beliefs: Optional[BeliefGraph] = None,
    ) -> None:
        super().__init__()
        self.ontology = ontology or Ontology()
        self.beliefs = beliefs
        if beliefs is not None:
            self._ingest_from_beliefs(beliefs)

    # ------------------------------------------------------------------
    def link(self, text: str, *, hint_type: Optional[str] = None) -> Dict[str, str]:
        """Resolve ``text`` into a canonical entity and type."""

        entity_type = hint_type or self._infer_type(text)
        canonical, resolved_type = self.resolve(text, entity_type=entity_type)
        return {"text": text, "canonical": canonical, "type": resolved_type}

    # ------------------------------------------------------------------
    def _infer_type(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return "Entity"
        if stripped.istitle() and " " in stripped:
            return "Person"
        if stripped and stripped[0].isupper():
            return "Place"
        return "Entity"

    def _ingest_from_beliefs(self, beliefs: BeliefGraph) -> None:
        try:
            for belief in beliefs.query(active_only=False):
                self.register(
                    belief.subject_label,
                    belief.subject_type,
                    canonical_id=belief.subject,
                    weight=0.01,
                )
                self.register(
                    belief.value_label,
                    belief.value_type,
                    canonical_id=belief.value,
                    weight=0.01,
                )
        except Exception:
            # Belief graph may not be initialised yet.
            return

