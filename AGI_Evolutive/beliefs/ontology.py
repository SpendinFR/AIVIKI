"""Lightweight ontology definitions for entity and relation types used by the belief graph."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set


logger = logging.getLogger(__name__)


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
        if parent and parent not in self.entity_types:
            logger.debug("Auto-registering missing parent entity '%s' for '%s'", parent, name)
            self.entity_types[parent] = EntityType(name=parent, parent="Entity" if parent != "Entity" else None)
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
    # Configuration helpers
    def load_from_mapping(
        self,
        config: Mapping[str, Any],
        *,
        clear_existing: bool = False,
    ) -> None:
        """Populate the ontology from a mapping structure.

        The mapping follows a lightweight schema::

            {
                "entities": [
                    {"name": "Agent", "parent": "Entity"},
                    "CustomEntity"
                ],
                "relations": [
                    {
                        "name": "knows",
                        "domain": ["Agent"],
                        "range": ["Agent"],
                        "polarity_sensitive": false,
                        "temporal": true,
                        "stability": "episode"
                    }
                ],
                "events": [
                    {
                        "name": "meeting",
                        "roles": {"host": ["Agent"]}
                    }
                ]
            }

        Missing optional fields fall back to sensible defaults, so the method
        is resilient to partial configurations.
        """

        if clear_existing:
            self.entity_types.clear()
            self.relation_types.clear()
            self.event_types.clear()

        for entity_entry in config.get("entities", []) or []:
            if isinstance(entity_entry, str):
                self.register_entity(entity_entry)
            else:
                name = entity_entry.get("name")
                if not name:
                    logger.warning("Skipping entity definition without a name: %s", entity_entry)
                    continue
                self.register_entity(name, parent=entity_entry.get("parent"))

        for relation_entry in config.get("relations", []) or []:
            if not isinstance(relation_entry, Mapping):
                logger.warning("Skipping invalid relation definition: %s", relation_entry)
                continue
            name = relation_entry.get("name")
            domain = relation_entry.get("domain", ["Entity"])
            range_ = relation_entry.get("range", ["Entity"])
            if not name:
                logger.warning("Skipping relation definition without a name: %s", relation_entry)
                continue
            self.register_relation(
                name,
                domain=domain,
                range=range_,
                polarity_sensitive=relation_entry.get("polarity_sensitive", True),
                temporal=relation_entry.get("temporal", False),
                stability=relation_entry.get("stability", "anchor"),
            )

        for event_entry in config.get("events", []) or []:
            if not isinstance(event_entry, Mapping):
                logger.warning("Skipping invalid event definition: %s", event_entry)
                continue
            name = event_entry.get("name")
            roles = event_entry.get("roles", {})
            if not name:
                logger.warning("Skipping event definition without a name: %s", event_entry)
                continue
            if not isinstance(roles, Mapping):
                logger.warning("Skipping event '%s' with invalid roles: %s", name, roles)
                continue
            self.register_event(name, roles={role: list(options) for role, options in roles.items()})

    def load_from_file(self, path: Path | str, *, clear_existing: bool = False) -> None:
        """Load ontology definitions from a JSON file."""

        data: Mapping[str, Any]
        raw_path = Path(path)
        with raw_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected a mapping at the top-level of {raw_path}")
        self.load_from_mapping(data, clear_existing=clear_existing)

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
        onto.register_entity("Context", parent="Entity")
        onto.register_entity("Emotion", parent="Entity")
        onto.register_entity("Goal", parent="Entity")
        onto.register_entity("Intention", parent="Goal")
        onto.register_entity("Resource", parent="Object")
        onto.register_entity("Tool", parent="Resource")
        onto.register_entity("Knowledge", parent="Entity")
        onto.register_entity("Experience", parent="Entity")
        onto.register_entity("Communication", parent="Activity")

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
        onto.register_relation(
            "located_in",
            domain=["Entity"],
            range=["Place"],
            stability="anchor",
        )
        onto.register_relation(
            "uses",
            domain=["Agent"],
            range=["Tool", "Resource"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "has_goal",
            domain=["Agent"],
            range=["Goal", "Intention"],
            stability="anchor",
        )
        onto.register_relation(
            "influences",
            domain=["Agent", "Organization", "Context"],
            range=["Agent", "Context", "Emotion"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "reports_on",
            domain=["Agent", "Organization"],
            range=["Knowledge", "Experience"],
            stability="anchor",
        )
        onto.register_relation(
            "expresses",
            domain=["Agent"],
            range=["Emotion", "Intention"],
            temporal=True,
            stability="episode",
        )
        onto.register_relation(
            "precedes",
            domain=["Activity", "TemporalSegment"],
            range=["Activity", "TemporalSegment"],
            temporal=True,
            stability="episode",
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
        onto.register_event(
            "collaboration",
            roles={
                "participants": ["Agent", "Organization"],
                "goal": ["Goal", "Intention"],
                "resource": ["Resource", "Tool"],
                "context": ["Context", "Place"],
            },
        )
        onto.register_event(
            "observation",
            roles={
                "observer": ["Agent"],
                "subject": ["Entity"],
                "insight": ["Knowledge", "Experience"],
            },
        )
        onto.register_event(
            "goal_progress",
            roles={
                "agent": ["Agent"],
                "goal": ["Goal", "Intention"],
                "milestone": ["TemporalSegment", "Experience"],
            },
        )
        return onto
