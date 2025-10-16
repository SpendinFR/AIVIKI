# -*- coding: utf-8 -*-
"""
Système de Créativité Avancée de l'AGI Évolutive
- Génération d'idées (divergente, latérale, transfert de domaines, etc.)
- Mélange conceptuel
- Détection d'insights
- Gestion de projets d'innovation
Implémente des gardes robustes + normalisation silencieuse pour éviter les erreurs de type
"string indices must be integers, not 'str'".
"""
from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque

try:
    import networkx as nx  # utilisé pour l'espace conceptuel
except Exception:  # fallback minimal si networkx indisponible
    class _MiniGraph(dict):
        def add_node(self, n, **attrs): self.setdefault(n, {}).update(attrs)
        def add_edge(self, a, b, **attrs):
            self.setdefault(a, {}); self.setdefault(b, {})
            self.setdefault("_edges", {}).setdefault((a,b), {}).update(attrs)
        def nodes(self, data=False):
            if data: return [(k, v) for k,v in self.items() if k!="_edges"]
            return [k for k in self.keys() if k!="_edges"]
        def neighbors(self, n):
            ed = self.get("_edges", {}); res = []
            for (a,b) in ed:
                if a==n: res.append(b)
                if b==n: res.append(a)
            return res
        def get_edge_data(self, a, b): return self.get("_edges", {}).get((a,b), {})
        def degree(self, n): return len(self.neighbors(n))
        def number_of_nodes(self): return len([k for k in self.keys() if k!="_edges"])
        def number_of_edges(self): return len(self.get("_edges", {}))
    nx = type("nx", (), {"Graph": _MiniGraph})

# -------------------- Types & Data --------------------

class IdeaState(Enum):
    RAW = "brute"
    DEVELOPED = "developpee"
    REFINED = "affinee"
    INTEGRATED = "integree"
    IMPLEMENTED = "implementee"
    ARCHIVED = "archivee"

class InsightType(Enum):
    ASSOCIATIVE = "associatif"
    RESTRUCTURING = "restructuration"
    ANALOGICAL = "analogique"
    EMERGENT = "emergent"
    INTUITIVE = "intuitif"

@dataclass
class CreativeIdea:
    id: str
    concept_core: str
    description: str
    state: IdeaState
    novelty: float
    usefulness: float
    feasibility: float
    elaboration: float
    domains: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    development_history: List[Dict[str, Any]] = field(default_factory=list)
    emotional_affinity: float = 0.3
    activation_level: float = 0.1

@dataclass
class ConceptualBlend:
    blend_id: str
    input_spaces: List[str]
    generic_space: List[str]
    blended_space: str
    emergent_structure: List[str]
    creativity_score: float
    coherence: float
    elaboration_potential: float

@dataclass
class CreativeInsight:
    insight_id: str
    type: InsightType
    content: str
    significance: float
    clarity: float
    surprise: float
    emotional_intensity: float
    preceding_incubation: float
    trigger: str
    timestamp: float
    related_ideas: List[str]
    verification_status: str

@dataclass
class InnovationProject:
    project_id: str
    core_idea: str
    objectives: List[str]
    constraints: List[str]
    resources_needed: List[str]
    development_phases: List[Dict[str, Any]]
    current_phase: int
    success_metrics: Dict[str, float]
    risk_assessment: Dict[str, float]
    team_dynamics: Dict[str, Any]

# -------------------- Utils & Guards --------------------

def _clip(v: float, lo: float=0.0, hi: float=1.0) -> float:
    try:
        v = float(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def _is_ci(x: Any) -> bool:
    return isinstance(x, CreativeIdea)

def _ensure_ci(x: Any, topic: str="") -> CreativeIdea:
    if isinstance(x, CreativeIdea):
        return x
    s = str(x)
    return CreativeIdea(
        id=f"idea_{int(time.time()*1000)}_{random.randint(100,999)}",
        concept_core=s,
        description=f"Idée: {s}",
        state=IdeaState.RAW,
        novelty=0.5, usefulness=0.5, feasibility=0.5, elaboration=0.3,
        domains=[], components=[], associations=[]
    )

def crea_normalize(self: "CreativitySystem") -> None:
    # Containers dict
    for name in ("creative_states","creative_history","idea_generation","conceptual_blending",
                 "insight_detection","innovation_engine","creative_processes",
                 "processing_threads","creative_knowledge","conceptual_space"):
        if not isinstance(getattr(self, name, None), dict):
            setattr(self, name, {})
    # Keys
    self.creative_states.setdefault("creative_flow", 0.3)
    self.creative_states.setdefault("cognitive_flexibility", 0.5)
    self.creative_states.setdefault("inspiration_level", 0.4)
    self.creative_states.setdefault("creative_confidence", 0.5)
    self.creative_history.setdefault("ideas_generated", deque(maxlen=5000))
    self.creative_history.setdefault("blends_created", deque(maxlen=1000))
    self.creative_history.setdefault("insights_experienced", deque(maxlen=500))
    self.creative_history.setdefault("projects_completed", deque(maxlen=100))
    self.creative_history.setdefault("creative_breakthroughs", deque(maxlen=50))
    self.creative_history.setdefault("learning_trajectory", deque(maxlen=1000))

    # idea_generation
    self.idea_generation.setdefault("idea_pool", deque(maxlen=1000))
    self.idea_generation.setdefault("generation_strategies", {})
    # coerce pool to CreativeIdea
    pool = list(self.idea_generation["idea_pool"])
    fixed = []
    for it in pool:
        fixed.append(_ensure_ci(it))
    self.idea_generation["idea_pool"].clear()
    self.idea_generation["idea_pool"].extend(fixed)

    # conceptual space and knowledge
    self.conceptual_space.setdefault("concept_network", nx.Graph())
    self.conceptual_blending.setdefault("blend_history", deque(maxlen=500))
    self.creative_knowledge.setdefault("analogical_sources", {})
    self.creative_knowledge.setdefault("heuristics", {})
    self.creative_knowledge.setdefault("constraint_templates", {})
    self.creative_knowledge.setdefault("innovation_patterns", {})

# -------------------- Subsystems (light) --------------------

class ActivationSpreadingSystem:
    def __init__(self, factor: float=0.3, decay: float=0.99):
        self.factor = factor
        self.decay = decay
    def tick(self, G):
        for node in G.nodes():
            act = G.nodes[node].get("activation", 0.1) * self.decay
            G.nodes[node]["activation"] = max(0.01, act)
    def boost(self, G, start: str, amount: float=0.2):
        if start in G.nodes():
            G.nodes[start]["activation"] = _clip(G.nodes[start].get("activation",0.1)+amount)
            for nb in getattr(G, "neighbors", lambda n: [])(start):
                w = G.get_edge_data(start, nb).get("weight", 0.5)
                G.nodes[nb]["activation"] = _clip(G.nodes[nb].get("activation",0.1)+w*self.factor*amount)

# -------------------- Main System --------------------

class CreativitySystem:
    """
    Système de créativité robuste, avec normalisation silencieuse et threads d'arrière-plan.
    """
    def __init__(self, cognitive_architecture: Any=None, memory_system: Any=None,
                 reasoning_system: Any=None, emotional_system: Any=None,
                 metacognitive_system: Any=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system or (getattr(cognitive_architecture, "memory", None) if cognitive_architecture else None)
        self.reasoning_system = reasoning_system or (getattr(cognitive_architecture, "reasoning", None) if cognitive_architecture else None)
        self.emotional_system = emotional_system or (getattr(cognitive_architecture, "emotions", None) if cognitive_architecture else None)
        self.metacognitive_system = metacognitive_system or (getattr(cognitive_architecture, "metacognition", None) if cognitive_architecture else None)

        # Core containers
        self.conceptual_space: Dict[str, Any] = {"concept_network": nx.Graph(), "act": ActivationSpreadingSystem()}
        self.idea_generation: Dict[str, Any] = {
            "idea_pool": deque(maxlen=1000),
            "generation_strategies": {},
        }
        self.conceptual_blending: Dict[str, Any] = {"blend_history": deque(maxlen=500)}
        self.insight_detection: Dict[str, Any] = {"insight_history": deque(maxlen=200)}
        self.innovation_engine: Dict[str, Any] = {"innovation_pipeline": deque(maxlen=50)}
        self.creative_processes: Dict[str, Any] = {"current_phase": "preparation", "phase_transitions": deque(maxlen=100)}
        self.creative_states: Dict[str, Any] = {
            "creative_flow": 0.3, "cognitive_flexibility": 0.5,
            "inspiration_level": 0.4, "creative_confidence": 0.5
        }
        self.contextual_influences: Dict[str, Any] = {
            "environmental_stimuli": [], "cultural_constraints": [], "domain_knowledge": {},
            "emotional_climate": 0.5, "cognitive_load": 0.3, "time_pressure": 0.2
        }
        self.creative_knowledge: Dict[str, Any] = {
            "analogical_sources": {
                "nature": ["évolution", "symbiose", "adaptation"],
                "technologie": ["réseau", "interface", "automatisation"],
                "art": ["composition", "contraste", "rythme"],
            },
            "heuristics": {},
            "constraint_templates": {},
            "innovation_patterns": {}
        }
        self.creative_history: Dict[str, Any] = {
            "ideas_generated": deque(maxlen=5000),
            "blends_created": deque(maxlen=1000),
            "insights_experienced": deque(maxlen=500),
            "projects_completed": deque(maxlen=100),
            "creative_breakthroughs": deque(maxlen=50),
            "learning_trajectory": deque(maxlen=1000),
        }
        self.processing_threads: Dict[str, Any] = {}
        self.running = True

        # Init
        crea_normalize(self)
        self._initialize_basic_conceptual_network()
        self._initialize_generation_strategies()
        self._initialize_innate_creativity()

        # Start background
        self._start_creative_monitoring()
        self._start_incubation_process()
        self._start_insight_detection()

        # First cycle
        self._initiate_first_creative_cycle()

    # -------------------- Initialization helpers --------------------

    def _initialize_basic_conceptual_network(self) -> None:
        G = self.conceptual_space["concept_network"]
        base = ["espace","temps","mouvement","énergie","forme","couleur","son","texture","quantité","qualité",
                "relation","causalité","similarité","différence","partie","tout","ordre","chaos","symétrie","asymétrie"]
        for c in base:
            G.add_node(c, activation=0.1, domain="fondamental")
        for a,b in [("espace","temps"),("mouvement","énergie"),("forme","couleur"),
                    ("partie","tout"),("ordre","chaos"),("symétrie","asymétrie"),
                    ("relation","causalité"),("similarité","différence")]:
            G.add_edge(a,b, weight=0.8, type="fondamental")

    def _initialize_generation_strategies(self) -> None:
        self.idea_generation["generation_strategies"] = {
            "random_association": self._strat_random_association,
            "domain_transfer": self._strat_domain_transfer,
            "constraint_challenge": self._strat_constraint_challenge,
            "attribute_listing": self._strat_attribute_listing,
            "forced_relationship": self._strat_forced_relationship,
        }

    def _initialize_innate_creativity(self) -> None:
        self.creative_knowledge["heuristics"].update({
            "analogy": {"description": "Similarités entre domaines", "effectiveness": 0.7},
            "combination": {"description": "Combiner des éléments", "effectiveness": 0.8},
            "transformation": {"description": "Transformer un concept", "effectiveness": 0.6},
        })
        self.creative_knowledge["innovation_patterns"].update({
            "problem_solution": {"description": "Problème → Solution", "success_rate": 0.6},
            "improvement": {"description": "Amélioration incrémentale", "success_rate": 0.7},
        })

    # -------------------- Background threads --------------------

    def _creative_monitoring_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._monitor_creative_state()
                self._update_conceptual_space()
                self._evaluate_ongoing_ideas()
            except Exception:
                pass
            time.sleep(2)

    def _incubation_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._process_incubation_phase()
            except Exception:
                pass
            time.sleep(10)

    def _insight_detection_loop(self) -> None:
        while self.running:
            try:
                crea_normalize(self)
                self._monitor_insight_conditions()
            except Exception:
                pass
            time.sleep(5)

    def _start_creative_monitoring(self) -> None:
        import threading

        t = threading.Thread(target=self._creative_monitoring_loop, daemon=True)
        t.start()
        self.processing_threads["creative_monitoring"] = t

    def _start_incubation_process(self) -> None:
        import threading

        t = threading.Thread(target=self._incubation_loop, daemon=True)
        t.start()
        self.processing_threads["incubation_process"] = t

    def _start_insight_detection(self) -> None:
        import threading

        t = threading.Thread(target=self._insight_detection_loop, daemon=True)
        t.start()
        self.processing_threads["insight_detection"] = t

    # -------------------- First cycle --------------------

    def _initiate_first_creative_cycle(self) -> None:
        crea_normalize(self)
        ideas = self.generate_ideas("créativité", constraints=[], num_ideas=5, strategy="random_association")
        self.creative_processes["current_phase"] = "preparation"
        self.creative_history["learning_trajectory"].append({
            "timestamp": time.time(),
            "event": "first_cycle",
            "ideas_generated": len(ideas),
            "creative_state": dict(self.creative_states),
        })

    # -------------------- Idea generation --------------------

    def generate_ideas(self, topic: str, constraints: List[str], num_ideas: int=10, strategy: str="auto") -> List[CreativeIdea]:
        crea_normalize(self)
        if strategy == "auto":
            strategy = self._select_strategy(topic, constraints)
        func = self.idea_generation["generation_strategies"].get(strategy, self._strat_random_association)
        raw = func(topic, constraints, num_ideas)
        out: List[CreativeIdea] = []
        for s in raw:
            ci = self._develop_raw_idea(s, topic, constraints)
            out.append(ci)
            self.idea_generation["idea_pool"].append(ci)
            self.creative_history["ideas_generated"].append(ci)
        self._update_creative_metrics(len(out), max(0.001, 0.05*len(out)), strategy)
        return out

    def _select_strategy(self, topic: str, constraints: List[str]) -> str:
        lvl = len(constraints)/10.0
        if lvl > 0.7: return "constraint_challenge"
        if " " not in topic and len(topic)<8: return "random_association"
        if random.random()<0.3: return "domain_transfer"
        return "attribute_listing"

    # Strategies (return list[str])
    def _strat_random_association(self, topic: str, constraints: List[str], n: int) -> List[str]:
        G = self.conceptual_space["concept_network"]
        nodes = list(G.nodes()) or [topic]
        ideas = []
        for _ in range(n*3):
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a==b and len(nodes)>1:
                b = random.choice([x for x in nodes if x!=a])
            ideas.append(f"{a} + {b} → {a}-{b} hybride")
            if len(ideas)>=n: break
        return ideas[:n]

    def _strat_domain_transfer(self, topic: str, constraints: List[str], n: int) -> List[str]:
        src = list(self.creative_knowledge["analogical_sources"].keys())
        ideas = []
        for _ in range(n):
            d = random.choice(src) if src else "source"
            c = random.choice(self.creative_knowledge["analogical_sources"].get(d, ["concept"]))
            ideas.append(f"Transférer {c} ({d}) vers {topic}")
        return ideas

    def _strat_constraint_challenge(self, topic: str, constraints: List[str], n: int) -> List[str]:
        ideas = []
        for i in range(max(1, min(n, len(constraints) or 1))):
            if constraints:
                c = constraints[i % len(constraints)]
                ideas.append(f"{topic} sans '{c}'")
            else:
                ideas.append(f"{topic} sans contrainte X")
        while len(ideas)<n:
            ideas.append(f"{topic} approche radicale #{len(ideas)+1}")
        return ideas[:n]

    def _strat_attribute_listing(self, topic: str, constraints: List[str], n: int) -> List[str]:
        attrs = ["taille","couleur","forme","texture","poids","durée","intensité","complexité","vitesse","coût"]
        mods = ["amplifier","réduire","inverser","combiner","transformer"]
        ideas = []
        for _ in range(n):
            ideas.append(f"{topic} avec {random.choice(attrs)} à {random.choice(mods)}")
        return ideas

    def _strat_forced_relationship(self, topic: str, constraints: List[str], n: int) -> List[str]:
        G = self.conceptual_space["concept_network"]
        nodes = list(G.nodes()) or [topic]
        random.shuffle(nodes)
        ideas = []
        for c in nodes[:n]:
            if c==topic: continue
            ideas.append(f"Relation forcée {topic} ↔ {c}")
        while len(ideas)<n: ideas.append(f"Relation forcée {topic} ↔ concept_{len(ideas)+1}")
        return ideas[:n]

    # Develop raw idea into CreativeIdea
    def _develop_raw_idea(self, raw: str, topic: str, constraints: List[str]) -> CreativeIdea:
        novelty = _clip(random.gauss(0.65, 0.2))
        usefulness = _clip(random.gauss(0.6, 0.2))
        feasibility = _clip(random.gauss(0.55, 0.2))
        elaboration = _clip(random.gauss(0.35, 0.15))
        return CreativeIdea(
            id=f"idea_{int(time.time()*1000)}_{random.randint(100,999)}",
            concept_core=str(raw),
            description=f"Idée créative: {raw}",
            state=IdeaState.RAW,
            novelty=novelty,
            usefulness=usefulness,
            feasibility=feasibility,
            elaboration=elaboration,
            domains=[],
            components=[],
            associations=[],
            emotional_affinity=_clip(random.random()),
            activation_level=0.1
        )

    # -------------------- Monitoring --------------------

    def _monitor_creative_state(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"])[-10:] if _is_ci(i)]
        if ideas:
            self.creative_states["creative_flow"] = _clip(self.creative_states.get("creative_flow",0.3) + 0.02*len(ideas))
            self.creative_states["creative_confidence"] = _clip(self.creative_states.get("creative_confidence",0.5) + 0.01*len(ideas))

    def _update_conceptual_space(self) -> None:
        G = self.conceptual_space["concept_network"]
        act = self.conceptual_space.get("act")
        if act:
            act.tick(G)

    def _evaluate_ongoing_ideas(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"])[-20:] if _is_ci(i)]
        for i in ideas:
            if i.state == IdeaState.RAW and random.random()<0.1:
                i.state = IdeaState.DEVELOPED
                i.last_modified = time.time()
                i.development_history.append({"timestamp": time.time(), "action": "promotion_auto", "state": "developed"})

    def _process_incubation_phase(self) -> None:
        ideas = [i for i in list(self.idea_generation["idea_pool"]) if _is_ci(i)]
        for i in ideas:
            if i.state in (IdeaState.RAW, IdeaState.DEVELOPED):
                i.activation_level = _clip(i.activation_level + random.uniform(0.01, 0.05))

    def _monitor_insight_conditions(self) -> None:
        cond = 0
        if self.creative_states.get("creative_flow",0.3) > 0.6: cond += 1
        if self.creative_states.get("cognitive_flexibility",0.5) > 0.5: cond += 1
        if self.creative_states.get("inspiration_level",0.4) > 0.4: cond += 1
        if 0.3 < self.contextual_influences.get("cognitive_load",0.5) < 0.7: cond += 1
        if cond >=3 and random.random()<0.1:
            self.experience_insight({"problem":"optimisation_processus_créatif"})

    # -------------------- Blending --------------------

    def create_conceptual_blend(self, concept1: str, concept2: str) -> ConceptualBlend:
        generic = ["relation","structure","processus"]
        blended = f"{concept1}-{concept2} hybride via {random.choice(generic)}"
        emergent = [f"structure émergente autour de {concept1} & {concept2}"]
        creativity = _clip(len(blended)/80.0)
        coherence = _clip(0.5 + 0.1*random.random())
        elabor = _clip(0.4 + 0.2*random.random())
        blend = ConceptualBlend(
            blend_id=f"blend_{int(time.time()*1000)}_{random.randint(100,999)}",
            input_spaces=[concept1, concept2],
            generic_space=generic,
            blended_space=blended,
            emergent_structure=emergent,
            creativity_score=creativity,
            coherence=coherence,
            elaboration_potential=elabor
        )
        self.conceptual_blending["blend_history"].append(blend)
        self.creative_history["blends_created"].append(blend)
        return blend

    # -------------------- Insights --------------------

    def experience_insight(self, problem_context: Dict[str, Any]) -> Optional[CreativeInsight]:
        content = f"Perspective nouvelle sur {problem_context.get('problem','un problème')}"
        t = time.time()
        insight = CreativeInsight(
            insight_id=f"insight_{int(t*1000)}_{random.randint(100,999)}",
            type=InsightType.EMERGENT,
            content=content,
            significance=_clip(random.uniform(0.5,0.9)),
            clarity=_clip(random.uniform(0.6,0.9)),
            surprise=_clip(random.uniform(0.6,0.9)),
            emotional_intensity=_clip(random.uniform(0.5,0.9)),
            preceding_incubation=random.uniform(10.0,60.0),
            trigger=random.choice(["changement de perspective","association distante","relaxation"]),
            timestamp=t,
            related_ideas=[i.id for i in list(self.idea_generation["idea_pool"])[-2:] if _is_ci(i)],
            verification_status="unverified"
        )
        self.insight_detection.setdefault("insight_history", deque(maxlen=200)).append(insight)
        self.creative_history["insights_experienced"].append(insight)
        self._update_after_insight(insight)
        return insight

    def _update_after_insight(self, insight: CreativeInsight) -> None:
        self.creative_states["creative_confidence"] = _clip(self.creative_states.get("creative_confidence",0.5) + 0.2*insight.significance)
        self.creative_states["inspiration_level"] = _clip(self.creative_states.get("inspiration_level",0.4) + 0.15*insight.emotional_intensity)

    # -------------------- Innovation --------------------

    def develop_innovation_project(self, core_idea: str, objectives: List[str], constraints: List[str]) -> InnovationProject:
        phases = [
            {"name":"exploration","duration_estimate":random.uniform(5,15),"resources":["recherche","brainstorming"]},
            {"name":"prototype","duration_estimate":random.uniform(10,30),"resources":["conception","test"]},
            {"name":"validation","duration_estimate":random.uniform(8,20),"resources":["expérimentation","feedback"]},
            {"name":"implémentation","duration_estimate":random.uniform(15,40),"resources":["déploiement","documentation"]},
        ]
        risks = {"technique": random.uniform(0.3,0.8), "marché": random.uniform(0.2,0.7), "ressources": _clip(len(constraints)/10.0), "temporel": random.uniform(0.4,0.9)}
        metrics = {"satisfaction_utilisateur": random.uniform(0.6,0.9), "impact_innovation": random.uniform(0.5,0.8)}
        res = ["temps","attention","énergie_cognitive"]
        project = InnovationProject(
            project_id=f"project_{int(time.time()*1000)}_{random.randint(100,999)}",
            core_idea=core_idea, objectives=objectives, constraints=constraints,
            resources_needed=res, development_phases=phases, current_phase=0,
            success_metrics=metrics, risk_assessment=risks,
            team_dynamics={"auto_collaboration": True, "roles":["générateur","évaluateur","développeur"]}
        )
        self.innovation_engine["innovation_pipeline"].append(project)
        self.creative_history["projects_completed"].append(project)
        return project

    # -------------------- Metrics & Status --------------------

    def _update_creative_metrics(self, ideas_generated: int, generation_time: float, strategy: str) -> None:
        eff = ideas_generated / max(generation_time, 1e-6)
        self.creative_states["creative_flow"] = _clip(self.creative_states.get("creative_flow",0.3) + 0.1 * min(0.2, 0.01*eff))
        if ideas_generated>0:
            self.creative_states["creative_confidence"] = _clip(self.creative_states.get("creative_confidence",0.5) + 0.02*ideas_generated)
        self.creative_history["learning_trajectory"].append({
            "timestamp": time.time(),
            "strategy": strategy,
            "ideas_generated": ideas_generated,
            "generation_time": generation_time,
            "efficiency": eff,
            "new_state": dict(self.creative_states)
        })

    def get_creative_status(self) -> Dict[str, Any]:
        pool = [i for i in list(self.idea_generation["idea_pool"])[-10:] if _is_ci(i)]
        return {
            "creative_states": dict(self.creative_states),
            "current_phase": self.creative_processes.get("current_phase","preparation"),
            "recent_activity": {
                "ideas_generated": len(pool),
                "ideas_by_state": { s.value: sum(1 for i in pool if i.state==s) for s in IdeaState }
            },
            "conceptual_space": {
                "concepts_count": self.conceptual_space["concept_network"].number_of_nodes(),
                "connections_count": self.conceptual_space["concept_network"].number_of_edges(),
            },
            "innovation_pipeline": {
                "projects_count": len(self.innovation_engine.get("innovation_pipeline", [])),
            },
            "historical_metrics": {
                "total_ideas": len(self.creative_history["ideas_generated"]),
                "total_blends": len(self.creative_history["blends_created"]),
                "total_insights": len(self.creative_history["insights_experienced"]),
                "total_projects": len(self.creative_history["projects_completed"]),
                "breakthroughs": len(self.creative_history["creative_breakthroughs"]),
            }
        }

    def stop_creativity_system(self) -> None:
        self.running = False

__all__ = ["CreativitySystem","CreativeIdea","ConceptualBlend","CreativeInsight","InnovationProject","IdeaState","InsightType"]
