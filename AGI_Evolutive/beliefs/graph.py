from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import os, json, time, uuid

@dataclass
class Evidence:
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
class Belief:
    id: str
    subject: str
    relation: str
    value: str
    confidence: float = 0.5      # croyance globale 0..1
    polarity: int = +1           # +1 assertion, -1 négation (contradictions gérées)
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    justifications: List[Evidence] = field(default_factory=list)
    created_by: str = "system"
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def new(subject: str, relation: str, value: str, *, confidence: float = 0.5, polarity: int = +1, created_by: str = "system") -> "Belief":
        return Belief(
            id=str(uuid.uuid4()),
            subject=str(subject),
            relation=str(relation),
            value=str(value),
            confidence=float(max(0.0, min(1.0, confidence))),
            polarity=+1 if polarity >= 0 else -1,
            valid_from=time.time(),
            valid_to=None,
            justifications=[],
            created_by=created_by,
            updated_at=time.time(),
        )

class BeliefGraph:
    """Stockage JSONL (append-only) + index mémoire, gestion contradictions et mise à jour 'AGM-lite'."""
    def __init__(self, path: str = "data/beliefs.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._cache: Dict[str, Belief] = {}
        self._load()

    def _load(self) -> None:
        self._cache.clear()
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    b = Belief(
                        id=obj["id"],
                        subject=obj["subject"],
                        relation=obj["relation"],
                        value=obj["value"],
                        confidence=obj.get("confidence", 0.5),
                        polarity=obj.get("polarity", +1),
                        valid_from=obj.get("valid_from", 0.0),
                        valid_to=obj.get("valid_to", None),
                        justifications=[Evidence(**e) for e in obj.get("justifications", [])],
                        created_by=obj.get("created_by", "system"),
                        updated_at=obj.get("updated_at", time.time()),
                    )
                    self._cache[b.id] = b
                except Exception:
                    continue

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for b in self._cache.values():
                row = asdict(b)
                row["justifications"] = [asdict(e) for e in b.justifications]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------- API ----------------
    def all(self) -> List[Belief]:
        return list(self._cache.values())

    def query(self, *, subject: Optional[str]=None, relation: Optional[str]=None, value: Optional[str]=None,
              min_conf: float=0.0, active_only: bool=True) -> List[Belief]:
        res = []
        now = time.time()
        for b in self._cache.values():
            if subject and b.subject != subject: continue
            if relation and b.relation != relation: continue
            if value and b.value != value: continue
            if b.confidence < min_conf: continue
            if active_only and b.valid_to and b.valid_to < now: continue
            res.append(b)
        return sorted(res, key=lambda x: (x.confidence, x.updated_at), reverse=True)

    def upsert(self, subject: str, relation: str, value: str, *,
               confidence: float, polarity: int=+1, evidence: Optional[Evidence]=None, created_by: str="system") -> Belief:
        # Si même triplet actif existe: combine (moyenne pondérée) + ajoute evidence
        # Si contradiction (polarity opposée) forte → abaisse l'autre croyance (AGM-lite)
        match = None
        for b in self.query(subject=subject, relation=relation, value=value, active_only=True):
            match = b; break
        if match:
            # update conf: moyenne entre ancienne conf et nouvelle, pondérée par poids evidence si fournie
            w = (evidence.weight if evidence else 0.5)
            match.confidence = float(max(0.0, min(1.0, (1-w)*match.confidence + w*confidence)))
            match.updated_at = time.time()
            if evidence: match.justifications.append(evidence)
            self._flush()
            return match

        # Contradiction?
        contradictions = [b for b in self.query(subject=subject, relation=relation, active_only=True) if b.value != value]
        for c in contradictions:
            if c.polarity != polarity and confidence > c.confidence:
                # la nouvelle évidence contredit plus sûrement: on décote l'ancienne
                c.confidence = float(max(0.0, c.confidence - 0.3*confidence))
                c.updated_at = time.time()
        b = Belief.new(subject, relation, value, confidence=confidence, polarity=polarity, created_by=created_by)
        if evidence: b.justifications.append(evidence)
        self._cache[b.id] = b
        self._flush()
        return b

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

    def summarize(self, subject: Optional[str]=None) -> Dict[str, Any]:
        bs = self.query(subject=subject) if subject else self.all()
        rels: Dict[str, List[Belief]] = {}
        for b in bs:
            rels.setdefault(b.relation, []).append(b)
        summary = {}
        for rel, items in rels.items():
            top = sorted(items, key=lambda x: x.confidence, reverse=True)[:5]
            summary[rel] = [{"value": i.value, "conf": i.confidence, "pol": i.polarity} for i in top]
        return summary
