"""Simple entity linker handling aliases and synonym resolution for the belief graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class EntityRecord:
    canonical_id: str
    name: str
    entity_type: str
    popularity: float = 0.5

    def bump(self, weight: float) -> None:
        self.popularity = min(1.0, self.popularity + weight)


class EntityLinker:
    """
    Maintains the low-level alias table for the belief graph.

    Ce composant ne dépend que des structures de croyances et sert de
    fondation à la façade ``knowledge.EntityLinker`` qui ajoute une
    intégration avec l'ontologie et la mémoire déclarative.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, EntityRecord] = {}
        self._aliases: Dict[str, str] = {}

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    # ------------------------------------------------------------------
    def register(self, name: str, entity_type: str, *, canonical_id: Optional[str] = None, weight: float = 0.1) -> str:
        """Registers an entity and returns its canonical identifier."""

        key = self._normalize(name)
        canonical_id = canonical_id or key
        record = self._entities.get(canonical_id)
        if record:
            record.bump(weight)
        else:
            record = EntityRecord(canonical_id=canonical_id, name=name, entity_type=entity_type, popularity=min(1.0, weight))
            self._entities[canonical_id] = record
        self._aliases[key] = canonical_id
        return canonical_id

    def alias(self, alias: str, canonical_id: str, weight: float = 0.05) -> None:
        if canonical_id not in self._entities:
            return
        self._aliases[self._normalize(alias)] = canonical_id
        self._entities[canonical_id].bump(weight)

    # ------------------------------------------------------------------
    def resolve(self, name: str, *, entity_type: Optional[str] = None) -> Tuple[str, str]:
        """Returns (canonical_id, entity_type)."""

        norm = self._normalize(name)
        canonical = self._aliases.get(norm)
        if canonical:
            record = self._entities[canonical]
            if entity_type and record.entity_type != entity_type:
                record.entity_type = entity_type
            return canonical, record.entity_type

        # new entity
        canonical_id = self.register(name, entity_type or "Entity", canonical_id=norm, weight=0.05)
        record = self._entities[canonical_id]
        if entity_type and record.entity_type != entity_type:
            record.entity_type = entity_type
        return canonical_id, record.entity_type

    # ------------------------------------------------------------------
    def merge(self, preferred: str, duplicate: str) -> None:
        """Fuses two entity identifiers, updating aliases."""

        preferred_norm = self._normalize(preferred)
        dup_norm = self._normalize(duplicate)
        if preferred_norm == dup_norm:
            return

        pref_id = self._aliases.get(preferred_norm, preferred_norm)
        dup_id = self._aliases.get(dup_norm, dup_norm)

        if dup_id == pref_id:
            return

        pref_record = self._entities.get(pref_id)
        dup_record = self._entities.pop(dup_id, None)
        if not pref_record or not dup_record:
            return

        pref_record.bump(dup_record.popularity)
        self._aliases[dup_norm] = pref_id
        for alias, canonical in list(self._aliases.items()):
            if canonical == dup_id:
                self._aliases[alias] = pref_id

    def get(self, canonical_id: str) -> Optional[EntityRecord]:
        return self._entities.get(canonical_id)

    def known_entities(self) -> Dict[str, EntityRecord]:
        return dict(self._entities)

    def canonical_form(self, name: str) -> str:
        norm = self._normalize(name)
        return self._aliases.get(norm, norm)
