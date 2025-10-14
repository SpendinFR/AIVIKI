# world_model/__init__.py
"""
World Model - module unique, optimisé et auto-contenu.
Contient :
  - PhysicsEngine        : dynamique physique simplifiée, contraintes et simulations
  - SocialModel          : agents, normes, intentions, relations
  - TemporalReasoning    : temps, causalité, échéances, fenêtres temporelles
  - SpatialReasoning     : repères, cartes mentales, graphes spatiaux, itinéraires

Objectifs :
  - Aucun import de sous-fichiers (évite ModuleNotFoundError)
  - Compatibilité avec core.cognitive_architecture (from world_model import PhysicsEngine)
  - to_state()/from_state() pour persistance (core/persistence.py)
  - Auto-wiring doux via cognitive_architecture (getattr, sans import croisé)

Dépendances : standard library only.
"""

from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 🧭 Utilitaires
# ============================================================

def _now() -> float:
    return time.time()

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


# ============================================================
# 🧩 SpatialReasoning
# ============================================================

@dataclass
class Node2D:
    id: str
    x: float
    y: float
    tags: List[str] = field(default_factory=list)


@dataclass
class Edge:
    a: str
    b: str
    cost: float


class SpatialReasoning:
    """
    Graphe spatial minimal : nœuds 2D + arêtes pondérées.
    - Construction de carte : add_node / add_edge
    - Itinéraire : shortest_path (Dijkstra simple)
    - Proximité : nearest_nodes
    - Repères : find_by_tag
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.nodes: Dict[str, Node2D] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self.landmarks: List[str] = []

    # ----- Carte & repères -----

    def add_node(self, node_id: str, x: float, y: float, tags: Optional[List[str]] = None):
        self.nodes[node_id] = Node2D(node_id, float(x), float(y), tags or [])
        if "landmark" in (tags or []):
            self.landmarks.append(node_id)

    def add_edge(self, a: str, b: str, cost: Optional[float] = None, bidirectional: bool = True):
        if a not in self.nodes or b not in self.nodes:
            return
        cost = float(cost) if cost is not None else self._euclid(a, b)
        self.edges.setdefault(a, []).append(Edge(a, b, cost))
        if bidirectional:
            self.edges.setdefault(b, []).append(Edge(b, a, cost))

    def _euclid(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    # ----- Requêtes -----

    def nearest_nodes(self, x: float, y: float, k: int = 3) -> List[Tuple[str, float]]:
        pairs = [(nid, math.hypot(n.x - x, n.y - y)) for nid, n in self.nodes.items()]
        pairs.sort(key=lambda t: t[1])
        return pairs[:max(0, k)]

    def find_by_tag(self, tag: str, k: int = 5) -> List[str]:
        return [nid for nid, n in self.nodes.items() if tag in n.tags][:k]

    # ----- Dijkstra minimal -----

    def shortest_path(self, start: str, goal: str) -> Tuple[float, List[str]]:
        if start not in self.nodes or goal not in self.nodes:
            return float("inf"), []
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        visited: set = set()
        frontier: List[Tuple[float, str]] = [(0.0, start)]
        import heapq
        while frontier:
            d, u = heapq.heappop(frontier)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                break
            for e in self.edges.get(u, []):
                v = e.b
                nd = d + e.cost
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(frontier, (nd, v))
        if goal not in dist:
            return float("inf"), []
        # reconstruct
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return dist[goal], path

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: {"x": n.x, "y": n.y, "tags": list(n.tags)} for nid, n in self.nodes.items()},
            "edges": {nid: [{"a": e.a, "b": e.b, "cost": e.cost} for e in lst] for nid, lst in self.edges.items()},
            "landmarks": list(self.landmarks),
        }

    def from_state(self, state: Dict[str, Any]):
        self.nodes = {}
        for nid, d in state.get("nodes", {}).items():
            self.nodes[nid] = Node2D(nid, float(d["x"]), float(d["y"]), list(d.get("tags", [])))
        self.edges = {}
        for nid, lst in state.get("edges", {}).items():
            self.edges[nid] = [Edge(x["a"], x["b"], float(x["cost"])) for x in lst]
        self.landmarks = list(state.get("landmarks", []))


# ============================================================
# ⏱️ TemporalReasoning
# ============================================================

@dataclass
class TimeWindow:
    start: float
    end: float
    label: str = ""


class TemporalReasoning:
    """
    Représentation simple du temps :
    - Fenêtres temporelles nommées
    - Délais/échéances (deadline pressure)
    - Raisonnement causal minimal (A avant B, délai, expiration)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.windows: List[TimeWindow] = []
        self.deadlines: Dict[str, float] = {}  # label -> timestamp
        self.causes: List[Tuple[str, str]] = []  # (cause, effect)

    def add_window(self, start: float, end: float, label: str):
        self.windows.append(TimeWindow(float(start), float(end), label))

    def set_deadline(self, label: str, timestamp: float):
        self.deadlines[label] = float(timestamp)

    def add_causal_link(self, cause: str, effect: str):
        self.causes.append((cause, effect))

    # ----- requêtes -----

    def is_within(self, t: float, label: Optional[str] = None) -> bool:
        for w in self.windows:
            if (label is None or w.label == label) and (w.start <= t <= w.end):
                return True
        return False

    def deadline_pressure(self, label: str, now: Optional[float] = None) -> float:
        now = _now() if now is None else float(now)
        ts = self.deadlines.get(label)
        if ts is None:
            return 0.0
        horizon = ts - now
        # pression augmente quand on s'approche de 0
        return _clamp(1.0 - (horizon / max(1.0, abs(horizon) + 1.0)), 0.0, 1.0)

    def can_happen_after(self, a: str, b: str) -> bool:
        # si (a -> b) causalement, b après a
        return (a, b) in self.causes

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "windows": [{"start": w.start, "end": w.end, "label": w.label} for w in self.windows],
            "deadlines": dict(self.deadlines),
            "causes": list(self.causes),
        }

    def from_state(self, state: Dict[str, Any]):
        self.windows = [TimeWindow(float(w["start"]), float(w["end"]), w.get("label", "")) for w in state.get("windows", [])]
        self.deadlines = {k: float(v) for k, v in state.get("deadlines", {}).items()}
        self.causes = [tuple(x) for x in state.get("causes", [])]


# ============================================================
# 🧑‍🤝‍🧑 SocialModel
# ============================================================

@dataclass
class Agent:
    id: str
    name: str
    traits: Dict[str, float] = field(default_factory=dict)    # ex: honesty, assertiveness
    affinity: Dict[str, float] = field(default_factory=dict)  # agent_id -> [-1..1]
    goals: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)


class SocialModel:
    """
    Modèle social minimal :
    - Agents, rôles, affinités
    - Normes sociales (règles simples)
    - Attribution d'intentions (Bayes léger)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.agents: Dict[str, Agent] = {}
        self.norms: List[Tuple[str, str, str]] = []  # (context, action, norm_label)
        self.intent_stats: Dict[str, Dict[str, float]] = {}  # agent -> intent -> weight

    # ----- Agents -----

    def add_agent(self, agent_id: str, name: str, traits: Optional[Dict[str, float]] = None):
        self.agents[agent_id] = Agent(agent_id, name, traits or {}, {}, [], [])

    def set_affinity(self, a: str, b: str, value: float):
        if a not in self.agents or b not in self.agents:
            return
        self.agents[a].affinity[b] = _clamp(float(value), -1.0, 1.0)

    def assign_role(self, agent_id: str, role: str):
        if agent_id in self.agents and role not in self.agents[agent_id].roles:
            self.agents[agent_id].roles.append(role)

    # ----- Normes -----

    def add_norm(self, context: str, action: str, label: str):
        self.norms.append((context, action, label))

    def applicable_norms(self, context: str) -> List[Tuple[str, str, str]]:
        return [n for n in self.norms if n[0] == context]

    # ----- Intentions (très simplifié) -----

    def update_intent(self, agent_id: str, intent: str, evidence: float):
        d = self.intent_stats.setdefault(agent_id, {})
        d[intent] = _clamp(d.get(intent, 0.5) + (evidence - 0.5) * 0.2, 0.0, 1.0)

    def most_likely_intent(self, agent_id: str) -> Optional[str]:
        d = self.intent_stats.get(agent_id, {})
        return max(d.items(), key=lambda kv: kv[1])[0] if d else None

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "agents": {
                aid: {
                    "name": a.name,
                    "traits": dict(a.traits),
                    "affinity": dict(a.affinity),
                    "goals": list(a.goals),
                    "roles": list(a.roles),
                }
                for aid, a in self.agents.items()
            },
            "norms": list(self.norms),
            "intent_stats": {k: dict(v) for k, v in self.intent_stats.items()},
        }

    def from_state(self, state: Dict[str, Any]):
        self.agents = {}
        for aid, d in state.get("agents", {}).items():
            self.agents[aid] = Agent(aid, d.get("name", aid), dict(d.get("traits", {})), dict(d.get("affinity", {})),
                                     list(d.get("goals", [])), list(d.get("roles", [])))
        self.norms = [tuple(x) for x in state.get("norms", [])]
        self.intent_stats = {k: dict(v) for k, v in state.get("intent_stats", {}).items()}


# ============================================================
# ⚙️ PhysicsEngine
# ============================================================

@dataclass
class Body:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    radius: float = 0.5
    friction: float = 0.02
    restitution: float = 0.2  # rebond 0..1
    tags: List[str] = field(default_factory=list)


class PhysicsEngine:
    """
    Moteur physique 2D discret et simplifié :
    - intégration d'Euler (dt fixe)
    - collisions disque-disque élastiques partiellement (restitution)
    - friction linéaire
    - contraintes spatiales (mur rectangulaire)
    - hooks d'observation vers cognition (reasoning/perception)
    """
    def __init__(self, cognitive_architecture: Any = None, memory_system: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.memory_system = memory_system
        self.bodies: Dict[str, Body] = {}
        self.bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0)  # xmin,ymin,xmax,ymax
        self.dt: float = 0.05
        self.last_step_ts: float = 0.0
        self.events: List[str] = []

        # auto-wiring
        ca = self.cognitive_arch
        if ca:
            self.reasoning = getattr(ca, "reasoning", None)
            self.perception = getattr(ca, "perception", None)
            self.goals = getattr(ca, "goals", None)
            self.metacognition = getattr(ca, "metacognition", None)
            # accès aux autres sous-modèles si exposés
            self.social = SocialModel(ca)
            self.temporal = TemporalReasoning(ca)
            self.spatial = SpatialReasoning(ca)
        else:
            self.social = SocialModel(None)
            self.temporal = TemporalReasoning(None)
            self.spatial = SpatialReasoning(None)

    # ----- Monde -----

    def add_body(self, body_id: str, x: float, y: float, **kwargs):
        self.bodies[body_id] = Body(body_id, float(x), float(y), **kwargs)

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.bounds = (float(xmin), float(ymin), float(xmax), float(ymax))

    # ----- Simulation -----

    def step(self, steps: int = 1):
        for _ in range(max(1, int(steps))):
            self._integrate()
            self._collisions()
        self.last_step_ts = _now()

    def _integrate(self):
        xmin, ymin, xmax, ymax = self.bounds
        for b in self.bodies.values():
            # friction
            b.vx *= (1.0 - b.friction)
            b.vy *= (1.0 - b.friction)
            # intégration
            b.x += b.vx * self.dt
            b.y += b.vy * self.dt
            # murs
            if b.x - b.radius < xmin or b.x + b.radius > xmax:
                b.vx *= -b.restitution
                b.x = _clamp(b.x, xmin + b.radius, xmax - b.radius)
                self.events.append(f"bounce_x::{b.id}")
            if b.y - b.radius < ymin or b.y + b.radius > ymax:
                b.vy *= -b.restitution
                b.y = _clamp(b.y, ymin + b.radius, ymax - b.radius)
                self.events.append(f"bounce_y::{b.id}")

    def _collisions(self):
        ids = list(self.bodies.keys())
        n = len(ids)
        for i in range(n):
            bi = self.bodies[ids[i]]
            for j in range(i + 1, n):
                bj = self.bodies[ids[j]]
                dx = bj.x - bi.x
                dy = bj.y - bi.y
                dist = math.hypot(dx, dy)
                min_dist = bi.radius + bj.radius
                if dist < min_dist and dist > 1e-6:
                    # normaliser
                    nx, ny = dx / dist, dy / dist
                    # correction de position simple
                    overlap = min_dist - dist
                    bi.x -= nx * overlap * 0.5
                    bi.y -= ny * overlap * 0.5
                    bj.x += nx * overlap * 0.5
                    bj.y += ny * overlap * 0.5
                    # échange de vitesse projetée sur la normale (restitution)
                    vi = bi.vx * nx + bi.vy * ny
                    vj = bj.vx * nx + bj.vy * ny
                    vi_new = vj * bi.restitution
                    vj_new = vi * bj.restitution
                    bi.vx += (vi_new - vi) * nx
                    bi.vy += (vi_new - vi) * ny
                    bj.vx += (vj_new - vj) * nx
                    bj.vy += (vj_new - vj) * ny
                    self.events.append(f"collide::{bi.id}::{bj.id}")

    # ----- Observations simplifiées -----

    def snapshot(self) -> Dict[str, Any]:
        return {
            "bodies": {k: vars(v) for k, v in self.bodies.items()},
            "bounds": self.bounds,
            "events": list(self.events[-50:]),
            "time": self.last_step_ts,
        }

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "bodies": {bid: {
                "x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy,
                "mass": b.mass, "radius": b.radius,
                "friction": b.friction, "restitution": b.restitution,
                "tags": list(b.tags),
            } for bid, b in self.bodies.items()},
            "bounds": tuple(self.bounds),
            "dt": self.dt,
            "events": list(self.events[-200:]),
            "temporal": self.temporal.to_state(),
            "social": self.social.to_state(),
            "spatial": self.spatial.to_state(),
        }

    def from_state(self, state: Dict[str, Any]):
        self.bodies = {}
        for bid, d in state.get("bodies", {}).items():
            self.bodies[bid] = Body(
                bid, float(d["x"]), float(d["y"]), float(d.get("vx", 0.0)), float(d.get("vy", 0.0)),
                float(d.get("mass", 1.0)), float(d.get("radius", 0.5)),
                float(d.get("friction", 0.02)), float(d.get("restitution", 0.2)),
                list(d.get("tags", []))
            )
        b = state.get("bounds", (-10.0, -10.0, 10.0, 10.0))
        self.bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        self.dt = float(state.get("dt", 0.05))
        self.events = list(state.get("events", []))
        self.temporal.from_state(state.get("temporal", {}))
        self.social.from_state(state.get("social", {}))
        self.spatial.from_state(state.get("spatial", {}))


# ============================================================
# 🔗 Exports publics
# ============================================================

__all__ = [
    "PhysicsEngine",
    "SocialModel",
    "TemporalReasoning",
    "SpatialReasoning",
]
