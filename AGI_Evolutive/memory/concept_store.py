from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import os
import json
import time
import math
import uuid

from AGI_Evolutive.utils.jsonsafe import json_sanitize


@dataclass
class Concept:
    id: str
    label: str
    support: float = 0.0
    recency: float = 1.0
    salience: float = 0.0
    confidence: float = 0.5
    last_seen: float = field(default_factory=time.time)
    examples: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class Relation:
    id: str
    src: str
    dst: str
    rtype: str
    weight: float = 0.0
    confidence: float = 0.5
    updated_at: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)


class ConceptStore:
    """Petite base graphe concepts/relations avec persistance JSON."""

    def __init__(self, path_concepts: str = "data/concepts.json", path_dashboard: str = "data/concepts_dashboard.json"):
        self.path_concepts = path_concepts
        self.path_dashboard = path_dashboard
        os.makedirs(os.path.dirname(self.path_concepts), exist_ok=True)
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relation] = {}
        self._load()

    def upsert_concept(
        self,
        label: str,
        support_delta: float,
        salience_delta: float,
        example_mem_id: Optional[str],
        confidence: float = 0.6,
    ) -> Concept:
        cid = self._find_by_label(label) or str(uuid.uuid4())[:8]
        concept = self.concepts.get(cid)
        if not concept:
            concept = Concept(id=cid, label=label, support=0.0, salience=0.0, confidence=confidence)
            self.concepts[cid] = concept
        now = time.time()
        decay = self._exp_decay(concept.last_seen, now, half_life=48 * 3600)
        concept.support = concept.support * decay + support_delta
        concept.salience = max(0.0, min(1.0, concept.salience * decay + salience_delta))
        concept.recency = 1.0
        concept.last_seen = now
        concept.confidence = max(concept.confidence, confidence)
        if example_mem_id and len(concept.examples) < 10:
            concept.examples.append(example_mem_id)
        return concept

    def upsert_relation(
        self,
        src_cid: str,
        dst_cid: str,
        rtype: str,
        weight_delta: float,
        mem_id: Optional[str],
        confidence: float = 0.6,
    ) -> Relation:
        rid = f"{src_cid}::{rtype}::{dst_cid}"
        relation = self.relations.get(rid)
        if not relation:
            relation = Relation(id=rid, src=src_cid, dst=dst_cid, rtype=rtype, weight=0.0, confidence=confidence)
            self.relations[rid] = relation
        now = time.time()
        decay = self._exp_decay(relation.updated_at, now, half_life=72 * 3600)
        relation.weight = relation.weight * decay + weight_delta
        relation.confidence = max(relation.confidence, confidence)
        relation.updated_at = now
        if mem_id and len(relation.evidence) < 20:
            relation.evidence.append(mem_id)
        return relation

    def get_top_concepts(self, k: int = 20) -> List[Concept]:
        return sorted(
            self.concepts.values(),
            key=lambda c: (0.6 * c.support + 0.4 * c.salience),
            reverse=True,
        )[:k]

    def neighbors(self, cid: str, rtype: Optional[str] = None, k: int = 10) -> List[Tuple[Relation, Concept]]:
        results: List[Tuple[Relation, Concept]] = []
        for relation in self.relations.values():
            if relation.src != cid:
                continue
            if rtype is not None and relation.rtype != rtype:
                continue
            dst = self.concepts.get(relation.dst)
            if dst:
                results.append((relation, dst))
        results.sort(key=lambda item: item[0].weight, reverse=True)
        return results[:k]

    def find_by_label_prefix(self, prefix: str, k: int = 10) -> List[Concept]:
        lower_prefix = prefix.lower()
        matches = [concept for concept in self.concepts.values() if concept.label.lower().startswith(lower_prefix)]
        matches.sort(key=lambda concept: concept.support, reverse=True)
        return matches[:k]

    def save(self) -> None:
        data = {
            "concepts": {cid: asdict(concept) for cid, concept in self.concepts.items()},
            "relations": {rid: asdict(relation) for rid, relation in self.relations.items()},
            "saved_at": time.time(),
        }
        with open(self.path_concepts, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(data), handle, ensure_ascii=False, indent=2)
        dashboard = {
            "t": time.time(),
            "top_concepts": [asdict(concept) for concept in self.get_top_concepts(25)],
            "counts": {"concepts": len(self.concepts), "relations": len(self.relations)},
        }
        with open(self.path_dashboard, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(dashboard), handle, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.path_concepts):
            return
        try:
            with open(self.path_concepts, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.concepts = {cid: Concept(**payload) for cid, payload in data.get("concepts", {}).items()}
            self.relations = {rid: Relation(**payload) for rid, payload in data.get("relations", {}).items()}
        except Exception:
            self.concepts = {}
            self.relations = {}

    def _find_by_label(self, label: str) -> Optional[str]:
        lower_label = label.lower()
        for cid, concept in self.concepts.items():
            if concept.label.lower() == lower_label:
                return cid
        return None

    @staticmethod
    def _exp_decay(last_t: float, now_t: float, half_life: float) -> float:
        dt = max(0.0, now_t - last_t)
        if half_life <= 0:
            return 1.0
        return math.pow(0.5, dt / half_life)
