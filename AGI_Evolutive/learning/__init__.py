
# learning/__init__.py
"""
Système d'Apprentissage Évolutif de l'AGI — module unique et auto‑contenu.

Ce fichier regroupe TOUTES les classes d’apprentissage en un seul module :
- ExperientialLearning (apprentissage expérientiel, cycle de Kolb)
- MetaLearning (méta-apprentissage, ajuste les hyper‑paramètres)
- TransferLearning (transfert inter-domaines, mapping analogique)
- ReinforcementLearning (apprentissage par renforcement tabulaire simple)
- CuriosityEngine (récompense intrinsèque, exploration)

Points clés :
- AUCUNE importation de sous-modules (évite l’erreur d’import).
- Méthodes to_state()/from_state() pour la persistance.
- Auto‑wiring sécurisé via getattr(self.cognitive_architecture, ...).
- Idempotent : si un sous-composant n’existe pas, le code reste stable.
"""

from __future__ import annotations
import time, math, random, hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 🌱 Utilitaires communs
# ============================================================

def _now() -> float:
    return time.time()

def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default

def _hash_str(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


# ============================================================
# 🧩 1) Apprentissage EXPÉRIENTIEL (cycle de Kolb)
# ============================================================

@dataclass
class LearningEpisode:
    id: str
    timestamp: float
    raw: Dict[str, Any] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotional_valence: float = 0.0
    confidence_gain: float = 0.0
    integration_level: float = 0.0


class ExperientialLearning:
    """
    Cycle complet : expérience concrète → observation réflexive → conceptualisation → expérimentation.
    Conçu pour intégrer des épisodes et enrichir la mémoire & les politiques.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        # journal d’expériences (persistant)
        self.learning_episodes: List[LearningEpisode] = []
        # compétences / états (adaptent les comportements)
        self.learning_competencies: Dict[str, float] = {
            "observation": 0.55,
            "reflection": 0.5,
            "abstraction": 0.45,
            "experimentation": 0.5,
            "pattern_detection": 0.6,
            "analogy": 0.5,
        }
        self.learning_states: Dict[str, float] = {
            "engagement": 0.65,
            "frustration_tolerance": 0.6,
            "openness": 0.75,
            "momentum": 0.5,
        }
        self.metrics: Dict[str, Any] = {
            "episodes_completed": 0,
            "concepts_formed": 0,
            "skills_compiled": 0,
            "errors_corrected": 0,
            "insights": 0,
        }
        self.learning_rate: float = 0.1

        # auto‑wiring (lecture uniquement, pas d’import croisé)
        ca = self.cognitive_architecture
        if ca:
            self.memory = getattr(ca, "memory", None)
            self.emotions = getattr(ca, "emotions", None)
            self.metacognition = getattr(ca, "metacognition", None)
            self.goals = getattr(ca, "goals", None)
            self.reasoning = getattr(ca, "reasoning", None)
            self.perception = getattr(ca, "perception", None)
            self.world_model = getattr(ca, "world_model", None)
            self.creativity = getattr(ca, "creativity", None)

    # --------- pipeline principal ---------

    def process_experience(self, raw: Dict[str, Any]) -> LearningEpisode:
        eid = f"ep::{int(_now())}::{_hash_str(str(raw))}"
        reflections = self._reflect(raw)
        concepts = self._abstract(raw, reflections)
        experiments = self._design_experiments(concepts)
        outcomes = self._execute_and_evaluate(experiments)
        emotional_valence = self._valence(outcomes)
        confidence_gain = self._confidence(outcomes)
        integration = self._integration(concepts, outcomes)

        episode = LearningEpisode(
            id=eid, timestamp=_now(), raw=raw, reflections=reflections,
            concepts=concepts, experiments=experiments, outcomes=outcomes,
            emotional_valence=emotional_valence, confidence_gain=confidence_gain,
            integration_level=integration
        )
        self.learning_episodes.append(episode)
        self.metrics["episodes_completed"] += 1
        self.metrics["concepts_formed"] += len(concepts)
        if outcomes:
            self.metrics["insights"] += 1

        # consolidation minimale en mémoire
        self._consolidate_episode(episode)
        # feedback à la méta‑cognition
        if getattr(self, "metacognition", None) and hasattr(self.metacognition, "register_learning_event"):
            try:
                self.metacognition.register_learning_event(episode.id, confidence_gain, integration)
            except Exception:
                pass
        return episode

    def _reflect(self, raw: Dict[str, Any]) -> List[str]:
        qs = [
            "Qu’est‑ce qui s’est réellement passé ?",
            "Qu’est‑ce que j’attendais ?",
            "Qu’est‑ce qui a surpris ?",
            "Qu’est‑ce que je dois vérifier ensuite ?",
        ]
        refs = [f"{q} → " + str(raw.get("summary", raw))[:120] for q in qs]
        # légère progression
        self.learning_competencies["reflection"] = min(1.0, self.learning_competencies["reflection"] + 0.01)
        return refs

    def _abstract(self, raw: Dict[str, Any], refs: List[str]) -> List[str]:
        concepts = []
        s = str(raw).lower()
        if "erreur" in s or "error" in s:
            concepts.append("Principe d’erreur : causes → effets (prévenir plutôt que corriger)")
        if "réussite" in s or "success" in s:
            concepts.append("Principe de réussite : répéter les conditions gagnantes")
        if "temps" in s or "delai" in s:
            concepts.append("Principe temporel : estimer/contraindre le temps utile")
        if not concepts:
            concepts.append("Principe de parcimonie : tester l’hypothèse la plus simple d’abord")
        self.learning_competencies["abstraction"] = min(1.0, self.learning_competencies["abstraction"] + 0.01)
        return concepts

    def _design_experiments(self, concepts: List[str]) -> List[Dict[str, Any]]:
        exps = []
        for c in concepts:
            exps.append({
                "concept": c,
                "type": "prediction_check",
                "risk": 0.3,
                "expected": "comportement cohérent avec le concept",
            })
            exps.append({
                "concept": c,
                "type": "generalization",
                "risk": 0.2,
                "expected": "fonctionne dans un contexte voisin",
            })
        self.learning_competencies["experimentation"] = min(1.0, self.learning_competencies["experimentation"] + 0.005)
        return exps

    def _execute_and_evaluate(self, exps: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        outcomes: Dict[str, Dict[str, float]] = {}
        for e in exps:
            risk = float(e.get("risk", 0.3))
            base = 0.7 - 0.4 * risk
            noise = random.uniform(-0.1, 0.1)
            success = max(0.05, min(0.95, base + noise))
            outcomes.setdefault(e["concept"], {"success_rate": 0.0, "learning_gain": 0.0})
            # agrégation simple
            outcomes[e["concept"]]["success_rate"] = (outcomes[e["concept"]]["success_rate"] + success) / 2 or success
            outcomes[e["concept"]]["learning_gain"] = (outcomes[e["concept"]]["learning_gain"] + (0.5 + success*0.3)) / 2
        return outcomes

    def _valence(self, outcomes: Dict[str, Dict[str, float]]) -> float:
        vals = [o.get("success_rate", 0.0) for o in outcomes.values()]
        return _safe_mean(vals, 0.5)

    def _confidence(self, outcomes: Dict[str, Dict[str, float]]) -> float:
        vals = [o.get("learning_gain", 0.0) for o in outcomes.values()]
        return _safe_mean(vals, 0.0) * 0.8

    def _integration(self, concepts: List[str], outcomes: Dict[str, Dict[str, float]]) -> float:
        if not concepts or not outcomes:
            return 0.0
        c = min(1.0, len(concepts) / 6.0)
        o = _safe_mean([x["success_rate"] for x in outcomes.values()], 0.5)
        return min(1.0, 0.4 * c + 0.6 * o)

    def _consolidate_episode(self, ep: LearningEpisode):
        mem = getattr(self, "memory", None)
        if not mem:
            return
        try:
            # stocke une trace épisodique simple (compatible avec la persistance maison)
            LTM = getattr(mem, "long_term_memory", {})
            key = f"learn::{ep.id}"
            if isinstance(LTM, dict):
                bucket = "EPISODIC"
                LTM.setdefault(bucket, {})[key] = {
                    "timestamp": ep.timestamp,
                    "concepts": ep.concepts,
                    "valence": ep.emotional_valence,
                    "confidence": ep.confidence_gain,
                    "integration": ep.integration_level,
                }
                if hasattr(mem, "memory_metadata") and isinstance(mem.memory_metadata, dict):
                    mem.memory_metadata["total_memories"] = mem.memory_metadata.get("total_memories", 0) + 1
        except Exception:
            pass

    # ---- API publique utile ----

    def to_state(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "learning_competencies": self.learning_competencies,
            "learning_states": self.learning_states,
            "metrics": self.metrics,
            "episodes": [ep.__dict__ for ep in self.learning_episodes[-200:]],  # limite snapshot
        }

    def from_state(self, state: Dict[str, Any]):
        self.learning_rate = state.get("learning_rate", 0.1)
        self.learning_competencies.update(state.get("learning_competencies", {}))
        self.learning_states.update(state.get("learning_states", {}))
        self.metrics.update(state.get("metrics", {}))
        self.learning_episodes = []
        for d in state.get("episodes", []):
            try:
                self.learning_episodes.append(LearningEpisode(**d))
            except Exception:
                continue

    def summarize(self) -> str:
        return f"{self.metrics.get('episodes_completed',0)} épisodes, {self.metrics.get('concepts_formed',0)} concepts."


# ============================================================
# 🧠 2) MÉTA‑APPRENTISSAGE
# ============================================================

class MetaLearning:
    """
    Ajuste dynamiquement les hyper‑paramètres d’apprentissage des autres composantes
    selon la performance agrégée (succès récents, confiance/integration).
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.performance: List[float] = []  # scores 0..1
        self.adjustment_rate: float = 0.05
        self.last_adjust_ts: float = 0.0

    def register_performance(self, score: float):
        self.performance.append(max(0.0, min(1.0, float(score))))
        if len(self.performance) > 200:
            self.performance.pop(0)
        self._maybe_adjust()

    def _maybe_adjust(self):
        now = _now()
        if now - self.last_adjust_ts < 5.0:
            return
        avg = _safe_mean(self.performance, 0.5)
        ca = self.cognitive_architecture
        if not ca:
            return
        # Ajuste le taux d’apprentissage expérientiel
        xl = getattr(ca, "learning", None)
        if xl and hasattr(xl, "learning_rate"):
            base = float(getattr(xl, "learning_rate", 0.1))
            delta = self.adjustment_rate * (0.5 - avg)
            setattr(xl, "learning_rate", max(0.01, min(1.0, base + delta)))
        # Ajuste aussi la curiosité si présente
        cur = getattr(ca, "curiosity", None) or getattr(ca, "learning", None)
        if cur and hasattr(cur, "curiosity_level"):
            cur.curiosity_level = max(0.1, min(1.0, cur.curiosity_level + (0.5 - avg) * 0.05))
        self.last_adjust_ts = now

    # aide à la persistance
    def to_state(self) -> Dict[str, Any]:
        return {"performance": self.performance, "last_adjust_ts": self.last_adjust_ts}

    def from_state(self, state: Dict[str, Any]):
        self.performance = state.get("performance", [])
        self.last_adjust_ts = state.get("last_adjust_ts", 0.0)


# ============================================================
# 🔄 3) TRANSFERT DE CONNAISSANCES
# ============================================================

@dataclass
class KnowledgeDomain:
    name: str
    concepts: Dict[str, Any]
    procedures: List[str] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)

class TransferLearning:
    """
    Cherche des analogies et mappings structurels entre domaines,
    pour réutiliser des concepts/procédures/principes.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.transfer_log: List[Dict[str, Any]] = []
        self.success_rate: float = 0.0

    def register_domain(self, domain_name: str, concepts: Dict[str, Any],
                        procedures: Optional[List[str]] = None,
                        principles: Optional[List[str]] = None) -> KnowledgeDomain:
        d = KnowledgeDomain(domain_name, concepts, procedures or [], principles or [])
        self.domains[domain_name] = d
        return d

    def _similarity(self, a: KnowledgeDomain, b: KnowledgeDomain) -> float:
        a_c, b_c = set(a.concepts.keys()), set(b.concepts.keys())
        if not a_c or not b_c:
            return 0.0
        inter = len(a_c & b_c)
        union = len(a_c | b_c)
        return inter / union

    def attempt_transfer(self, source: str, target: str, kinds: Optional[List[str]] = None) -> Dict[str, Any]:
        if source not in self.domains or target not in self.domains:
            raise ValueError("Domaines inconnus")
        src, tgt = self.domains[source], self.domains[target]
        sim = self._similarity(src, tgt)
        kinds = kinds or ["concepts", "procedures", "principles"]
        transferred: List[str] = []
        difficulty: List[str] = []

        if "concepts" in kinds:
            for k, v in src.concepts.items():
                if k not in tgt.concepts and sim > 0.3:
                    tgt.concepts[k] = {"adapted_from": source, "desc": str(v)[:200]}
                    transferred.append(f"concept::{k}")
                else:
                    difficulty.append(f"concept::{k}")
        if "procedures" in kinds:
            for p in src.procedures:
                if p not in tgt.procedures and sim > 0.2:
                    tgt.procedures.append(p + " (adapté)")
                    transferred.append(f"procedure::{p}")
                else:
                    difficulty.append(f"procedure::{p}")
        if "principles" in kinds:
            for pr in src.principles:
                if pr not in tgt.principles and sim > 0.1:
                    tgt.principles.append(pr + " (généralisé)")
                    transferred.append(f"principle::{pr}")
                else:
                    difficulty.append(f"principle::{pr}")

        overall_success = max(0.0, min(1.0, 0.5 * sim + 0.5 * (len(transferred) / max(1, len(transferred) + len(difficulty)))))
        self.transfer_log.append({
            "when": _now(),
            "source": source, "target": target,
            "similarity": sim, "success": overall_success,
            "transferred": transferred, "difficulty": difficulty,
        })
        # MAJ taux global
        self.success_rate = _safe_mean([t["success"] for t in self.transfer_log], 0.0)
        return self.transfer_log[-1]

    def to_state(self) -> Dict[str, Any]:
        return {
            "domains": {k: {"concepts": v.concepts, "procedures": v.procedures, "principles": v.principles} for k, v in self.domains.items()},
            "transfer_log": self.transfer_log,
            "success_rate": self.success_rate,
        }

    def from_state(self, state: Dict[str, Any]):
        self.domains = {}
        for name, d in state.get("domains", {}).items():
            self.domains[name] = KnowledgeDomain(name, d.get("concepts", {}), d.get("procedures", []), d.get("principles", []))
        self.transfer_log = state.get("transfer_log", [])
        self.success_rate = state.get("success_rate", 0.0)


# ============================================================
# 🏆 4) APPRENTISSAGE PAR RENFORCEMENT (tabulaire)
# ============================================================

class ReinforcementLearning:
    """
    Table de valeurs simple (state/action) avec mise à jour TD(0).
    Suffit pour moduler des choix locaux dans l’AGI.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.value_table: Dict[str, float] = {}
        self.alpha: float = 0.1
        self.gamma: float = 0.9
        self.last_state: Optional[str] = None
        self.last_action: Optional[str] = None

    def update_value(self, state: str, reward: float, next_state: Optional[str] = None) -> float:
        old = self.value_table.get(state, 0.0)
        next_val = self.value_table.get(next_state, 0.0) if next_state else 0.0
        new = old + self.alpha * (reward + self.gamma * next_val - old)
        self.value_table[state] = new
        self.last_state = state
        return new

    def choose_action(self, state: str, actions: List[str], eps: float = 0.2) -> str:
        if not actions:
            return ""
        if random.random() < eps:
            act = random.choice(actions)
        else:
            act = max(actions, key=lambda a: self.value_table.get(f"{state}|{a}", 0.0))
        self.last_action = act
        return act

    def to_state(self) -> Dict[str, Any]:
        return {"value_table": self.value_table, "alpha": self.alpha, "gamma": self.gamma}

    def from_state(self, state: Dict[str, Any]):
        self.value_table = state.get("value_table", {})
        self.alpha = float(state.get("alpha", 0.1))
        self.gamma = float(state.get("gamma", 0.9))


# ============================================================
# 🔭 5) MOTEUR DE CURIOSITÉ (récompense intrinsèque)
# ============================================================

class CuriosityEngine:
    """
    Génère des récompenses intrinsèques basées sur la nouveauté et l’imprévu.
    Peut stimuler la recherche d’information et l’auto-questionnement.
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.curiosity_level: float = 0.5
        self.seen_hashes: Dict[str, int] = {}
        self.history: List[Tuple[float, str, float]] = []  # (time, stimulus, reward)

    def _novelty(self, stimulus: str) -> float:
        h = _hash_str(stimulus, 8)
        c = self.seen_hashes.get(h, 0)
        self.seen_hashes[h] = c + 1
        # nouveauté: décroît avec répétition
        return 1.0 / (1.0 + c)

    def stimulate(self, stimulus: str) -> float:
        n = self._novelty(stimulus)
        reward = self.curiosity_level * n
        self.history.append((_now(), stimulus[:120], reward))
        # hook facultatif : booster la motivation des goals si présent
        ca = self.cognitive_architecture
        if ca and getattr(ca, "goals", None) and hasattr(ca.goals, "motivation_system"):
            try:
                ca.goals.motivation_system.boost_motivation(reward)
            except Exception:
                pass
        return reward

    def adjust(self, success_rate: float):
        self.curiosity_level = max(0.1, min(1.0, self.curiosity_level + (0.5 - success_rate) * 0.05))

    def to_state(self) -> Dict[str, Any]:
        return {"curiosity_level": self.curiosity_level, "seen_hashes": self.seen_hashes, "history": self.history[-200:]}

    def from_state(self, state: Dict[str, Any]):
        self.curiosity_level = float(state.get("curiosity_level", 0.5))
        self.seen_hashes = dict(state.get("seen_hashes", {}))
        self.history = list(state.get("history", []))


# ============================================================
# 🔗 Exports publics pour import direct: from learning import X
# ============================================================

__all__ = [
    "ExperientialLearning",
    "MetaLearning",
    "TransferLearning",
    "ReinforcementLearning",
    "CuriosityEngine",
]
