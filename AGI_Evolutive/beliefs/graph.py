from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Iterable, Tuple
import os, json, time, uuid, math

from AGI_Evolutive.utils.jsonsafe import json_sanitize

from .ontology import Ontology
from .entity_linker import EntityLinker

@dataclass
class Evidence:
    """
    Trace factuelle stockée avec une croyance.

    Les justifications manipulées ici sont persistées dans le graphe de
    croyances.  Elles sont plus riches (id, source, snippet, poids) que
    les "evidence" du module ``reasoning.structures`` qui ne servent qu'à
    journaliser une session d'inférence.
    """
    id: str
    kind: str                # "observation" | "dialog" | "memory" | "file" | "reasoning"
    source: str              # libre: "user", "inbox:<file>", "self", ...
    snippet: str
    weight: float = 0.5      # 0..1
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def new(kind: str, source: str, snippet: str, weight: float = 0.5) -> "Evidence":
        return Evidence(id=str(uuid.uuid4()), kind=kind, source=source, snippet=snippet[:500], weight=float(max(0.0, min(1.0, weight))))


@dataclass
class TemporalSegment:
    start: float
    end: Optional[float] = None
    recurrence: Optional[Dict[str, Any]] = None

    def is_active(self, timestamp: Optional[float] = None) -> bool:
        ts = timestamp or time.time()
        if ts < self.start:
            return False
        if self.end and ts > self.end:
            return False
        if not self.recurrence:
            return True

        tm = time.gmtime(ts)
        days = self.recurrence.get("days_of_week")
        if days is not None and tm.tm_wday not in days:
            return False

        minutes = tm.tm_hour * 60 + tm.tm_min
        start_minutes = self.recurrence.get("start_minutes")
        end_minutes = self.recurrence.get("end_minutes")
        if start_minutes is not None and minutes < start_minutes:
            return False
        if end_minutes is not None and minutes >= end_minutes:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "recurrence": self.recurrence,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "TemporalSegment":
        recurrence = payload.get("recurrence")
        if recurrence and isinstance(recurrence, dict):
            rec = dict(recurrence)
            for key in ("start_time", "end_time"):
                if key in rec and isinstance(rec[key], str):
                    hh, mm = rec[key].split(":")
                    minutes = int(hh) * 60 + int(mm)
                    if key == "start_time":
                        rec["start_minutes"] = minutes
                    else:
                        rec["end_minutes"] = minutes
            recurrence = rec
        return TemporalSegment(
            start=float(payload.get("start", time.time())),
            end=payload.get("end"),
            recurrence=recurrence,
        )


@dataclass
class Belief:
    id: str
    subject: str
    subject_label: str
    subject_type: str
    relation: str
    relation_type: str
    value: str
    value_label: str
    value_type: str
    confidence: float = 0.5      # croyance globale 0..1
    polarity: int = +1           # +1 assertion, -1 négation (contradictions gérées)
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    justifications: List[Evidence] = field(default_factory=list)
    created_by: str = "system"
    updated_at: float = field(default_factory=time.time)
    temporal_segments: List[TemporalSegment] = field(default_factory=list)
    stability: str = "anchor"

    @staticmethod
    def new(
        subject: str,
        relation: str,
        value: str,
        *,
        subject_label: Optional[str] = None,
        subject_type: str = "Entity",
        relation_type: str = "related_to",
        value_label: Optional[str] = None,
        value_type: str = "Entity",
        confidence: float = 0.5,
        polarity: int = +1,
        created_by: str = "system",
        temporal_segments: Optional[List[TemporalSegment]] = None,
        stability: str = "anchor",
    ) -> "Belief":
        return Belief(
            id=str(uuid.uuid4()),
            subject=str(subject),
            subject_label=str(subject_label or subject),
            subject_type=subject_type,
            relation=str(relation),
            relation_type=relation_type,
            value=str(value),
            value_label=str(value_label or value),
            value_type=value_type,
            confidence=float(max(0.0, min(1.0, confidence))),
            polarity=+1 if polarity >= 0 else -1,
            valid_from=time.time(),
            valid_to=None,
            justifications=[],
            created_by=created_by,
            updated_at=time.time(),
            temporal_segments=list(temporal_segments or []),
            stability=stability,
        )

    def is_active(self, timestamp: Optional[float] = None) -> bool:
        ts = timestamp or time.time()
        if self.valid_to and ts > self.valid_to:
            return False
        if ts < self.valid_from:
            return False
        if not self.temporal_segments:
            return True
        return any(seg.is_active(ts) for seg in self.temporal_segments)

    def decay(self, *, now: Optional[float] = None, rate: float = 0.0) -> None:
        now = now or time.time()
        elapsed = max(0.0, now - self.updated_at)
        if rate <= 0.0 or elapsed <= 0.0:
            return
        self.confidence = float(max(0.0, self.confidence * math.exp(-rate * elapsed)))
        if self.confidence < 0.01:
            self.valid_to = now

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["justifications"] = [asdict(e) for e in self.justifications]
        row["temporal_segments"] = [seg.to_dict() for seg in self.temporal_segments]
        row["kind"] = "belief"
        return row

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Belief":
        temporal_segments = [TemporalSegment.from_dict(seg) for seg in payload.get("temporal_segments", [])]
        return Belief(
            id=payload["id"],
            subject=payload.get("subject", ""),
            subject_label=payload.get("subject_label", payload.get("subject", "")),
            subject_type=payload.get("subject_type", "Entity"),
            relation=payload.get("relation", ""),
            relation_type=payload.get("relation_type", "related_to"),
            value=payload.get("value", ""),
            value_label=payload.get("value_label", payload.get("value", "")),
            value_type=payload.get("value_type", "Entity"),
            confidence=float(payload.get("confidence", 0.5)),
            polarity=int(payload.get("polarity", 1)),
            valid_from=float(payload.get("valid_from", 0.0)),
            valid_to=payload.get("valid_to"),
            justifications=[Evidence(**e) for e in payload.get("justifications", [])],
            created_by=payload.get("created_by", "system"),
            updated_at=float(payload.get("updated_at", time.time())),
            temporal_segments=temporal_segments,
            stability=payload.get("stability", "anchor"),
        )


@dataclass
class Event:
    id: str
    event_type: str
    roles: Dict[str, str]
    role_labels: Dict[str, str]
    occurred_at: float
    location: Optional[str]
    confidence: float
    justifications: List[Evidence] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def new(
        event_type: str,
        *,
        roles: Dict[str, str],
        role_labels: Optional[Dict[str, str]] = None,
        occurred_at: Optional[float] = None,
        location: Optional[str] = None,
        confidence: float = 0.5,
        evidences: Optional[List[Evidence]] = None,
    ) -> "Event":
        return Event(
            id=str(uuid.uuid4()),
            event_type=event_type,
            roles=dict(roles),
            role_labels=dict(role_labels or roles),
            occurred_at=occurred_at or time.time(),
            location=location,
            confidence=float(max(0.0, min(1.0, confidence))),
            justifications=list(evidences or []),
            updated_at=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "event",
            "id": self.id,
            "event_type": self.event_type,
            "roles": self.roles,
            "role_labels": self.role_labels,
            "occurred_at": self.occurred_at,
            "location": self.location,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "justifications": [asdict(e) for e in self.justifications],
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Event":
        return Event(
            id=payload["id"],
            event_type=payload.get("event_type", "event"),
            roles=dict(payload.get("roles", {})),
            role_labels=dict(payload.get("role_labels", payload.get("roles", {}))),
            occurred_at=float(payload.get("occurred_at", time.time())),
            location=payload.get("location"),
            confidence=float(payload.get("confidence", 0.5)),
            justifications=[Evidence(**e) for e in payload.get("justifications", [])],
            updated_at=float(payload.get("updated_at", time.time())),
        )


@dataclass
class LocalRule:
    id: str
    if_relation: str
    then_relation: str
    strength: float = 0.4
    polarity: int = +1
    subject_type: Optional[str] = None
    value_type: Optional[str] = None
    confidence_cap: float = 0.9

    def applies(self, belief: Belief) -> bool:
        if belief.relation != self.if_relation:
            return False
        if self.subject_type and belief.subject_type != self.subject_type:
            return False
        if self.value_type and belief.value_type != self.value_type:
            return False
        return True

    def project(self, belief: Belief) -> Tuple[str, str, str, int]:
        return belief.subject, self.then_relation, belief.value, self.polarity

class BeliefGraph:
    """Stockage JSONL (append-only) + index mémoire, gestion contradictions et mise à jour 'AGM-lite'."""

    def __init__(
        self,
        path: str = "data/beliefs.jsonl",
        *,
        ontology: Optional[Ontology] = None,
        entity_linker: Optional[EntityLinker] = None,
    ) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.ontology = ontology or Ontology.default()
        self.entity_linker = entity_linker or EntityLinker()
        self._cache: Dict[str, Belief] = {}
        self._events: Dict[str, Event] = {}
        self._rules: List[LocalRule] = self._default_rules()
        self.decay_rates = {"anchor": 0.0001, "episode": 0.001}
        self._load()

    def set_entity_linker(self, linker: EntityLinker) -> None:
        self.entity_linker = linker
        linker.beliefs = self

    def flush(self) -> None:
        self._flush()

    def iter_beliefs(self) -> Iterable[Belief]:
        return list(self._cache.values())

    def find_contradictions(self, min_conf: float = 0.6) -> List[Tuple[Belief, Belief]]:
        contradictions: List[Tuple[Belief, Belief]] = []
        index: Dict[Tuple[str, str], Belief] = {}
        for belief in self._cache.values():
            if belief.confidence < min_conf:
                continue
            key = (belief.subject, belief.relation)
            if belief.polarity < 0:
                positive = index.get(key)
                if positive and positive.value == belief.value:
                    contradictions.append((positive, belief))
            else:
                index[key] = belief
        return contradictions

    def _load(self) -> None:
        self._cache.clear()
        self._events.clear()
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    kind = obj.get("kind", "belief")
                    if kind == "event":
                        ev = Event.from_dict(obj)
                        self._events[ev.id] = ev
                        self._register_event_entities(ev)
                        continue

                    b = Belief.from_dict(obj)
                    self._normalize_belief(b)
                    self._cache[b.id] = b
                    self._register_belief_entities(b)
                except Exception:
                    continue

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for b in self._cache.values():
                f.write(json.dumps(json_sanitize(b.to_dict()), ensure_ascii=False) + "\n")
            for ev in self._events.values():
                f.write(json.dumps(json_sanitize(ev.to_dict()), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def _default_rules(self) -> List[LocalRule]:
        return [
            LocalRule(
                id="habit_implies_preference",
                if_relation="does_often",
                then_relation="likes",
                strength=0.6,
                subject_type="Agent",
                value_type="Activity",
                confidence_cap=0.85,
            ),
            LocalRule(
                id="opposition_dampens_preference",
                if_relation="opposes",
                then_relation="likes",
                strength=0.5,
                polarity=-1,
                confidence_cap=0.8,
            ),
            LocalRule(
                id="cause_implies_temporal",
                if_relation="causes",
                then_relation="temporal",
                strength=0.3,
                confidence_cap=0.7,
            ),
        ]

    def _normalize_belief(self, belief: Belief) -> None:
        relation_def = self.ontology.infer_relation_type(belief.relation)
        belief.relation_type = relation_def.name
        if not belief.stability:
            belief.stability = relation_def.stability
        if not belief.subject_type or belief.subject_type == "Entity":
            if relation_def.domain:
                belief.subject_type = next(iter(relation_def.domain))
        if not belief.value_type or belief.value_type == "Entity":
            if relation_def.range:
                belief.value_type = next(iter(relation_def.range))
        if belief.temporal_segments:
            belief.temporal_segments = [seg if isinstance(seg, TemporalSegment) else TemporalSegment.from_dict(seg) for seg in belief.temporal_segments]

    def _register_belief_entities(self, belief: Belief) -> None:
        self.entity_linker.register(belief.subject_label, belief.subject_type, canonical_id=belief.subject, weight=0.02)
        if belief.subject_label != belief.subject:
            self.entity_linker.alias(belief.subject_label, belief.subject, weight=0.01)
        self.entity_linker.register(belief.value_label, belief.value_type, canonical_id=belief.value, weight=0.02)
        if belief.value_label != belief.value:
            self.entity_linker.alias(belief.value_label, belief.value, weight=0.01)

    def _register_event_entities(self, event: Event) -> None:
        for role, canonical in event.roles.items():
            label = event.role_labels.get(role, canonical)
            record = self.entity_linker.get(canonical)
            entity_type = record.entity_type if record else "Entity"
            self.entity_linker.register(label, entity_type, canonical_id=canonical, weight=0.01)
            if label != canonical:
                self.entity_linker.alias(label, canonical, weight=0.005)

    def _find_belief(
        self,
        subject: str,
        relation: str,
        value: str,
        *,
        polarity: Optional[int] = None,
        active_only: bool = True,
        timestamp: Optional[float] = None,
    ) -> Optional[Belief]:
        ts = timestamp or time.time()
        for belief in self._cache.values():
            if belief.subject != subject or belief.relation != relation or belief.value != value:
                continue
            if polarity is not None and belief.polarity != polarity:
                continue
            if active_only and not belief.is_active(ts):
                continue
            return belief
        return None

    def _apply_rules(self, seed: Belief) -> List[Belief]:
        updates: List[Belief] = []
        now = time.time()
        for rule in self._rules:
            if not rule.applies(seed):
                continue
            if seed.polarity <= 0:
                continue
            derived_conf = min(rule.confidence_cap, seed.confidence * rule.strength)
            if derived_conf <= 0.0:
                continue
            relation = rule.then_relation
            relation_def = self.ontology.infer_relation_type(relation)
            if rule.polarity >= 0:
                existing = self._find_belief(seed.subject, relation, seed.value, polarity=+1, active_only=False)
                if existing:
                    if derived_conf > existing.confidence:
                        existing.confidence = derived_conf
                        existing.updated_at = now
                        updates.append(existing)
                    continue
                subject_record = self.entity_linker.get(seed.subject)
                value_record = self.entity_linker.get(seed.value)
                new_b = Belief.new(
                    subject=seed.subject,
                    subject_label=subject_record.name if subject_record else seed.subject_label,
                    subject_type=seed.subject_type,
                    relation=relation,
                    relation_type=relation_def.name,
                    value=seed.value,
                    value_label=value_record.name if value_record else seed.value_label,
                    value_type=seed.value_type,
                    confidence=derived_conf,
                    polarity=+1,
                    created_by=f"rule:{rule.id}",
                    temporal_segments=seed.temporal_segments,
                    stability=relation_def.stability,
                )
                new_b.justifications.append(
                    Evidence.new("reasoning", f"rule:{rule.id}", f"Dérivé de {seed.id}", weight=derived_conf)
                )
                self._cache[new_b.id] = new_b
                self._register_belief_entities(new_b)
                updates.append(new_b)
            else:
                positive = self._find_belief(seed.subject, relation, seed.value, polarity=+1, active_only=False)
                if positive:
                    positive.confidence = float(max(0.0, positive.confidence - derived_conf))
                    positive.updated_at = now
                    updates.append(positive)
                negative = self._find_belief(seed.subject, relation, seed.value, polarity=-1, active_only=False)
                if negative:
                    if derived_conf > negative.confidence:
                        negative.confidence = derived_conf
                        negative.updated_at = now
                        updates.append(negative)
                    continue
                subject_record = self.entity_linker.get(seed.subject)
                value_record = self.entity_linker.get(seed.value)
                new_b = Belief.new(
                    subject=seed.subject,
                    subject_label=subject_record.name if subject_record else seed.subject_label,
                    subject_type=seed.subject_type,
                    relation=relation,
                    relation_type=relation_def.name,
                    value=seed.value,
                    value_label=value_record.name if value_record else seed.value_label,
                    value_type=seed.value_type,
                    confidence=derived_conf,
                    polarity=-1,
                    created_by=f"rule:{rule.id}",
                    temporal_segments=seed.temporal_segments,
                    stability=relation_def.stability,
                )
                new_b.justifications.append(
                    Evidence.new("reasoning", f"rule:{rule.id}", f"Contradiction dérivée de {seed.id}", weight=derived_conf)
                )
                self._cache[new_b.id] = new_b
                self._register_belief_entities(new_b)
                updates.append(new_b)
        return updates

    # ---------------- API ----------------
    def all(self) -> List[Belief]:
        return sorted(self._cache.values(), key=lambda b: b.updated_at, reverse=True)

    def events(self) -> List[Event]:
        return sorted(self._events.values(), key=lambda ev: ev.occurred_at, reverse=True)

    def query(self, *, subject: Optional[str]=None, relation: Optional[str]=None, value: Optional[str]=None,
              min_conf: float=0.0, active_only: bool=True, subject_type: Optional[str]=None,
              relation_type: Optional[str]=None, value_type: Optional[str]=None, stability: Optional[str]=None,
              at_time: Optional[float]=None) -> List[Belief]:
        res = []
        now = at_time or time.time()
        subject_key = self.entity_linker.canonical_form(subject) if subject else None
        value_key = self.entity_linker.canonical_form(value) if value else None
        for b in self._cache.values():
            if subject_key and b.subject != subject_key and b.subject_label != subject: continue
            if relation and b.relation != relation: continue
            if relation_type and b.relation_type != relation_type: continue
            if value_key and b.value != value_key and b.value_label != value: continue
            if subject_type and b.subject_type != subject_type: continue
            if value_type and b.value_type != value_type: continue
            if stability and b.stability != stability: continue
            if b.confidence < min_conf: continue
            if active_only and not b.is_active(now): continue
            res.append(b)
        return sorted(res, key=lambda x: (x.confidence, x.updated_at), reverse=True)

    def query_events(
        self,
        *,
        event_type: Optional[str] = None,
        role: Optional[str] = None,
        entity: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[Event]:
        entity_key = self.entity_linker.canonical_form(entity) if entity else None
        items: List[Event] = []
        for ev in self._events.values():
            if event_type and ev.event_type != event_type:
                continue
            if since and ev.occurred_at < since:
                continue
            if until and ev.occurred_at > until:
                continue
            if role and entity_key:
                if ev.roles.get(role) != entity_key and ev.role_labels.get(role) != entity:
                    continue
            elif entity_key:
                if entity_key not in ev.roles.values() and entity not in ev.role_labels.values():
                    continue
            items.append(ev)
        return sorted(items, key=lambda ev: ev.occurred_at, reverse=True)

    def upsert(
        self,
        subject: str,
        relation: str,
        value: str,
        *,
        confidence: float,
        polarity: int = +1,
        evidence: Optional[Evidence] = None,
        created_by: str = "system",
        subject_type: Optional[str] = None,
        value_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        temporal_segments: Optional[Iterable[Any]] = None,
        stability: Optional[str] = None,
    ) -> Belief:
        relation_def = self.ontology.infer_relation_type(relation)
        relation_type = relation_type or relation_def.name
        stability = stability or relation_def.stability
        subject_type = subject_type or (next(iter(relation_def.domain)) if relation_def.domain else "Entity")
        value_type = value_type or (next(iter(relation_def.range)) if relation_def.range else "Entity")

        self.ontology.infer_entity_type(subject_type)
        self.ontology.infer_entity_type(value_type)

        subject_id, subject_type = self.entity_linker.resolve(subject, entity_type=subject_type)
        value_id, value_type = self.entity_linker.resolve(value, entity_type=value_type)

        if not relation_def.allows(subject_type, value_type, entities=self.ontology.entity_types):
            relation_def = self.ontology.infer_relation_type("related_to")
            relation_type = relation_def.name

        segments: List[TemporalSegment] = []
        if temporal_segments:
            for seg in temporal_segments:
                if isinstance(seg, TemporalSegment):
                    segments.append(seg)
                elif isinstance(seg, dict):
                    segments.append(TemporalSegment.from_dict(seg))

        conf = float(max(0.0, min(1.0, confidence)))
        now = time.time()
        match = self._find_belief(subject_id, relation, value_id, polarity=polarity, active_only=True)
        if match:
            w = evidence.weight if evidence else 0.5
            match.confidence = float(max(0.0, min(1.0, (1 - w) * match.confidence + w * conf)))
            match.updated_at = now
            if segments:
                match.temporal_segments = segments
            match.stability = stability or match.stability
            if evidence:
                match.justifications.append(evidence)
            self._register_belief_entities(match)
            self._apply_rules(match)
            self._flush()
            return match

        for c in list(self._cache.values()):
            if c.subject != subject_id or c.relation != relation:
                continue
            if not c.is_active(now):
                continue
            if c.value == value_id:
                continue
            if c.polarity != polarity and conf > c.confidence:
                c.confidence = float(max(0.0, c.confidence - 0.3 * conf))
                c.updated_at = now

        new_belief = Belief.new(
            subject=subject_id,
            subject_label=subject,
            subject_type=subject_type,
            relation=relation,
            relation_type=relation_type,
            value=value_id,
            value_label=value,
            value_type=value_type,
            confidence=conf,
            polarity=polarity,
            created_by=created_by,
            temporal_segments=segments,
            stability=stability,
        )
        if evidence:
            new_belief.justifications.append(evidence)
        self._normalize_belief(new_belief)
        self._cache[new_belief.id] = new_belief
        self._register_belief_entities(new_belief)
        self._apply_rules(new_belief)
        self._flush()
        return new_belief

    def add_evidence(self, belief_id: str, evidence: Evidence) -> bool:
        b = self._cache.get(belief_id)
        if not b: return False
        b.justifications.append(evidence); b.updated_at = time.time()
        # ajustement léger : plus d’évidences → confiance monte un peu (capée)
        b.confidence = float(min(1.0, b.confidence + 0.05*evidence.weight))
        self._flush()
        return True

    def retire(self, belief_id: str) -> bool:
        b = self._cache.get(belief_id)
        if not b: return False
        b.valid_to = time.time()
        b.updated_at = time.time()
        self._flush()
        return True

    def reactivate(self, belief_id: str, weight: float = 0.1) -> bool:
        b = self._cache.get(belief_id)
        if not b:
            return False
        b.confidence = float(min(1.0, b.confidence + weight))
        if b.valid_to and b.confidence > 0.3:
            b.valid_to = None
        b.updated_at = time.time()
        self._flush()
        return True

    def decay(self, *, now: Optional[float] = None) -> None:
        now = now or time.time()
        dirty = False
        for b in list(self._cache.values()):
            rate = self.decay_rates.get(b.stability, 0.0)
            previous = b.confidence
            b.decay(now=now, rate=rate)
            if b.confidence != previous:
                dirty = True
        if dirty:
            self._flush()

    def add_event(
        self,
        event_type: str,
        *,
        roles: Dict[str, str],
        role_labels: Optional[Dict[str, str]] = None,
        occurred_at: Optional[float] = None,
        location: Optional[str] = None,
        confidence: float = 0.5,
        evidences: Optional[List[Evidence]] = None,
    ) -> Event:
        schema = self.ontology.event(event_type)
        canonical_roles: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        for role, entity_name in roles.items():
            allowed_types = schema.roles.get(role) if schema else None
            entity_type = next(iter(allowed_types)) if allowed_types else "Entity"
            canonical, entity_type = self.entity_linker.resolve(entity_name, entity_type=entity_type)
            canonical_roles[role] = canonical
            labels[role] = (role_labels or {}).get(role, entity_name)
        event = Event.new(
            event_type,
            roles=canonical_roles,
            role_labels=labels,
            occurred_at=occurred_at,
            location=location,
            confidence=confidence,
            evidences=evidences,
        )
        self._events[event.id] = event
        self._register_event_entities(event)
        self._flush()
        return event

    def summarize(self, subject: Optional[str]=None) -> Dict[str, Any]:
        bs = self.query(subject=subject) if subject else self.all()
        rels: Dict[str, List[Belief]] = {}
        for b in bs:
            rels.setdefault(b.relation, []).append(b)
        summary = {}
        for rel, items in rels.items():
            top = sorted(items, key=lambda x: x.confidence, reverse=True)[:5]
            summary[rel] = [
                {
                    "value": i.value_label,
                    "value_id": i.value,
                    "conf": i.confidence,
                    "pol": i.polarity,
                    "stability": i.stability,
                    "temporal": [seg.to_dict() for seg in i.temporal_segments],
                }
                for i in top
            ]
        return summary
