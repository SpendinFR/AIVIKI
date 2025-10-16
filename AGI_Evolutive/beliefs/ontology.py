"""Lightweight ontology definitions for entity and relation types used by the belief graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class EntityType:
    """Represents a semantic type for an entity."""

    name: str
    parent: Optional[str] = None

    def is_a(self, other: str, *, registry: Dict[str, "EntityType"]) -> bool:
        if self.name == other:
            return True
        parent = self.parent
        while parent:
            if parent == other:
                return True
            parent = registry.get(parent).parent if registry.get(parent) else None
        return False


@dataclass(frozen=True)
class RelationType:
    """Represents the schema for a relation in the belief graph."""

    name: str
    domain: Set[str]
    range: Set[str]
    polarity_sensitive: bool = True
    temporal: bool = False
    stability: str = "anchor"  # "anchor" or "episode"

    def allows(self, subject_type: str, object_type: str, *, entities: Dict[str, EntityType]) -> bool:
        return any(
            entities.get(subject_type, EntityType(subject_type)).is_a(domain, registry=entities)
            for domain in self.domain
        ) and any(
            entities.get(object_type, EntityType(object_type)).is_a(rng, registry=entities)
            for rng in self.range
        )


@dataclass(frozen=True)
class EventType:
    """Schema for n-ary events."""

    name: str
    roles: Dict[str, Set[str]]  # role -> allowed entity types

    def validate_roles(self, assignments: Dict[str, str], *, entities: Dict[str, EntityType]) -> bool:
        for role, entity_type in assignments.items():
            allowed = self.roles.get(role)
            if not allowed:
                return False
            if not any(
                entities.get(entity_type, EntityType(entity_type)).is_a(option, registry=entities)
                for option in allowed
            ):
                return False
        return True


class Ontology:
    """Central registry for entity, relation and event types.

    C'est l'implémentation canonique côté ``beliefs`` ; le package
    ``knowledge`` se contente d'en fournir une enveloppe qui clone les
    types par défaut pour un usage read-only.
    """

    def __init__(self) -> None:
        self.entity_types: Dict[str, EntityType] = {}
        self.relation_types: Dict[str, RelationType] = {}
        self.event_types: Dict[str, EventType] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    def register_entity(self, name: str, *, parent: Optional[str] = None) -> None:
        self.entity_types[name] = EntityType(name=name, parent=parent)

    def register_relation(
        self,
        name: str,
        *,
        domain: Iterable[str],
        range: Iterable[str],
        polarity_sensitive: bool = True,
        temporal: bool = False,
        stability: str = "anchor",
    ) -> None:
        self.relation_types[name] = RelationType(
            name=name,
            domain=set(domain),
            range=set(range),
            polarity_sensitive=polarity_sensitive,
            temporal=temporal,
            stability=stability,
        )

    def register_event(self, name: str, *, roles: Dict[str, Iterable[str]]) -> None:
        self.event_types[name] = EventType(
            name=name,
            roles={role: set(options) for role, options in roles.items()},
        )

    # ------------------------------------------------------------------
    # Lookup helpers
    def entity(self, name: str) -> Optional[EntityType]:
        return self.entity_types.get(name)

    def relation(self, name: str) -> Optional[RelationType]:
        return self.relation_types.get(name)

    def event(self, name: str) -> Optional[EventType]:
        return self.event_types.get(name)

    # ------------------------------------------------------------------
    def infer_relation_type(self, name: str) -> RelationType:
        rel = self.relation(name)
        if rel:
            return rel
        # Fallback relation for unknown entries
        fallback = RelationType(
            name=name,
            domain={"Entity"},
            range={"Entity"},
            polarity_sensitive=True,
            temporal=False,
            stability="anchor",
        )
        self.relation_types[name] = fallback
        return fallback

    def infer_entity_type(self, name: str) -> EntityType:
        ent = self.entity(name)
        if ent:
            return ent
        fallback = EntityType(name=name, parent="Entity")
        self.entity_types[name] = fallback
        return fallback

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "Ontology":
        onto = cls()
        # Base hierarchy
        onto.register_entity("Entity")
        onto.register_entity("Agent", parent="Entity")
        onto.register_entity("Person", parent="Agent")
        onto.register_entity("Organization", parent="Agent")
        onto.register_entity("Place", parent="Entity")
        onto.register_entity("Object", parent="Entity")
        onto.register_entity("Food", parent="Object")
        onto.register_entity("Habit", parent="Entity")
        onto.register_entity("Activity", parent="Entity")
        onto.register_entity("TemporalSegment", parent="Entity")

        # Relations
        onto.register_relation("likes", domain=["Agent"], range=["Entity"], stability="anchor")
        onto.register_relation(
            "does_often",
            domain=["Agent"],
            range=["Activity"],
            temporal=True,
            stability="anchor",
        )
        onto.register_relation(
            "causes",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )
        onto.register_relation(
            "part_of",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )
        onto.register_relation(
            "opposes",
            domain=["Entity"],
            range=["Entity"],
            stability="episode",
        )
        onto.register_relation(
            "temporal",
            domain=["Activity", "TemporalSegment"],
            range=["TemporalSegment", "Activity"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "related_to",
            domain=["Entity"],
            range=["Entity"],
            stability="anchor",
        )

        # Events
        onto.register_event(
            "interaction",
            roles={
                "actor": ["Agent"],
                "target": ["Agent", "Entity"],
                "medium": ["Object", "Activity"],
                "location": ["Place"],
            },
        )
        onto.register_event(
            "consumption",
            roles={"actor": ["Agent"], "item": ["Food", "Object"], "location": ["Place"]},
        )
        onto.register_event(
            "routine",
            roles={"actor": ["Agent"], "activity": ["Activity"], "timeslot": ["TemporalSegment"]},
        )
        return onto
