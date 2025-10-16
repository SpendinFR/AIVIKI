# Représentation robuste d'un "pattern d’interaction" comme règle sociale testable
# Forme : ⟨ Contexte → Tactique → Effets_attendus ⟩ + incertitude (postérieurs Beta par effet)
# Aucun appel LLM. 100% JSON-sérialisable. N’abîme pas votre architecture.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import time, math, hashlib, json

# ---------- Petites bases ----------
TS = float

def _now() -> TS:
    return time.time()

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))

# ---------- Prédicats symboliques de contexte ----------
@dataclass
class Predicate:
    """
    Prédicat symbolique: "dialogue_act == insinuation", "risk_level <= medium", etc.
    ops supportés: 'eq','neq','in','nin','ge','gt','le','lt','exists','missing'
    """
    key: str
    op: str
    value: Any = None
    weight: float = 1.0  # importance dans le match global

    def test(self, ctx: Dict[str, Any]) -> Tuple[bool, float]:
        v = ctx
        for k in self.key.split("."):
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                v = None
                break

        ok = False
        if self.op == "exists":
            ok = v is not None
        elif self.op == "missing":
            ok = v is None
        elif self.op == "eq":
            ok = (v == self.value)
        elif self.op == "neq":
            ok = (v != self.value)
        elif self.op == "in":
            ok = (v in (self.value or []))
        elif self.op == "nin":
            ok = (v not in (self.value or []))
        elif self.op == "ge":
            try: ok = float(v) >= float(self.value)
            except: ok = False
        elif self.op == "gt":
            try: ok = float(v) > float(self.value)
            except: ok = False
        elif self.op == "le":
            try: ok = float(v) <= float(self.value)
            except: ok = False
        elif self.op == "lt":
            try: ok = float(v) < float(self.value)
            except: ok = False

        # score partiel = weight si ok, sinon 0
        return ok, (self.weight if ok else 0.0)

# ---------- Tactique (transformation pragmatique) ----------
@dataclass
class TacticSpec:
    """
    Tactique abstraite (pas un texte), paramétrable.
    Exemples: 'reformulation_empathique', 'banter_leger', 'question_socratique', 'ack_gracieux'
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.name}:{json.dumps(self.params, sort_keys=True, ensure_ascii=False)}"

# ---------- Effets observables & postérieurs ----------
@dataclass
class EffectPosterior:
    """
    Modélise un effet attendu avec incertitude via une Beta(α,β).
    Exemple d’effets: 'reduce_uncertainty', 'continue_dialogue', 'positive_valence', 'acceptance_marker'
    """
    alpha: float = 1.0
    beta: float = 1.0
    # Fenêtre glissante approximative via "decay" (0..1); ex: 0.98 → oublie en douceur
    decay: float = 0.995

    def expected(self) -> float:
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    def conf_int95(self) -> Tuple[float, float]:
        # Approximation rapide : ±1.96 * sqrt(p(1-p)/(n+3)) (n ≈ alpha+beta-2)
        n = max(0.0, self.alpha + self.beta - 2.0)
        p = self.expected()
        if n <= 0: return (max(0.0, p-0.5), min(1.0, p+0.5))
        import math
        se = math.sqrt(p * (1 - p) / (n + 3.0))
        return (clamp(p - 1.96 * se), clamp(p + 1.96 * se))

    def observe(self, success: bool):
        # Décroissance douce pour oublier le très ancien
        self.alpha = 1.0 + (self.alpha - 1.0) * self.decay
        self.beta  = 1.0 + (self.beta  - 1.0) * self.decay
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

# ---------- Règle sociale testable ----------
@dataclass
class InteractionRule:
    """
    Représente ⟨ Contexte → Tactique → Effets_attendus ⟩ + incertitude
    - context_predicates: liste de Predicate à matcher sur le contexte courant
    - tactic: TacticSpec
    - effects: dict nom_effet -> EffectPosterior
    - stats: usage_count, last_used_ts, confidence, ema_reward (aggrégat interne)
    - provenance: comment/depuis où (inbox:..., curated:..., mined:...)
    - tags: ex. ["mined","social","human-like"]
    """
    id: str
    version: str = "1.0"
    context_predicates: List[Predicate] = field(default_factory=list)
    tactic: TacticSpec = field(default_factory=lambda: TacticSpec("noop", {}))
    effects: Dict[str, EffectPosterior] = field(default_factory=dict)
    created_ts: TS = field(default_factory=_now)
    last_used_ts: TS = 0.0
    usage_count: int = 0
    confidence: float = 0.5    # synthèse (ex: moyenne des expected() d’effets clés)
    ema_reward: float = 0.5    # score agrégé multi-sources (Social Critic), 0..1
    cooldown: float = 0.0      # anti-abus probabiliste (utilisé par le sélecteur, pas ici)
    provenance: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=lambda: ["social"])

    # ----------- Usine de construction -----------
    @staticmethod
    def build(predicates: List[Predicate],
              tactic: TacticSpec,
              effects_names: Optional[List[str]] = None,
              provenance: Optional[Dict[str, Any]] = None) -> "InteractionRule":
        effects = {}
        for name in (effects_names or ["reduce_uncertainty", "continue_dialogue", "positive_valence", "acceptance_marker"]):
            effects[name] = EffectPosterior()
        rid = _hash(tactic.key() + "|" + "|".join(f"{p.key}:{p.op}:{p.value}" for p in predicates))
        return InteractionRule(
            id=rid,
            context_predicates=predicates,
            tactic=tactic,
            effects=effects,
            provenance=provenance or {}
        )

    # ----------- Matching contexte -----------
    def match_score(self, ctx: Dict[str, Any]) -> float:
        """
        Score de pertinence [0..1] basé sur les prédicats.
        On pondère par weight et normalise.
        """
        if not self.context_predicates:
            return 0.5  # neutre si aucune contrainte
        total_w = sum(max(0.0, p.weight) for p in self.context_predicates) or 1.0
        got = 0.0
        for p in self.context_predicates:
            ok, sc = p.test(ctx)
            got += sc
        return clamp(got / total_w)

    # ----------- Utilité attendue -----------
    def expected_utility(self, weights: Optional[Dict[str, float]] = None,
                         exploration_bonus: float = 0.0) -> float:
        """
        Utilité attendue d’appliquer la tactique, combinant effets (postérieurs),
        confiance synthétique, et un petit bonus d’exploration si désiré.
        """
        W = {
            "reduce_uncertainty": 0.35,
            "continue_dialogue":  0.25,
            "positive_valence":   0.25,
            "acceptance_marker":  0.15
        }
        if weights:
            W.update(weights)

        u = 0.0
        for name, w in W.items():
            post = self.effects.get(name)
            if post:
                u += w * post.expected()
            else:
                u += w * 0.5  # neutre si inconnu

        # synthèse de "confidence" : moyenne des expected()
        self.confidence = sum((self.effects.get(k, EffectPosterior()).expected() for k in W.keys())) / len(W)

        # exploration: bonus léger quand on n'a pas encore beaucoup essayé
        n = max(1, self.usage_count)
        bonus = exploration_bonus * (1.0 / math.sqrt(n))
        return clamp(u + bonus)

    # ----------- Cycle de vie -----------
    def register_use(self):
        self.usage_count += 1
        self.last_used_ts = _now()

    def observe_outcome(self, outcome: Dict[str, Any]):
        """
        Mise à jour des postérieurs d’effets à partir d’un 'outcome' symbolique.
        outcome: {
            "reduced_uncertainty": bool,
            "continued": bool,
            "valence": float (-1..+1),
            "accepted": bool,
            "reward": float (0..1),   # Social Critic agrégé
        }
        """
        # reduce_uncertainty
        if "reduced_uncertainty" in outcome:
            self.effects.setdefault("reduce_uncertainty", EffectPosterior()).observe(bool(outcome["reduced_uncertainty"]))
        # continue_dialogue
        if "continued" in outcome:
            self.effects.setdefault("continue_dialogue", EffectPosterior()).observe(bool(outcome["continued"]))
        # positive_valence (seuil 0)
        if "valence" in outcome:
            self.effects.setdefault("positive_valence", EffectPosterior()).observe(float(outcome["valence"]) > 0.0)
        # acceptance_marker (ex: “ok, merci”, “d’accord”)
        if "accepted" in outcome:
            self.effects.setdefault("acceptance_marker", EffectPosterior()).observe(bool(outcome["accepted"]))

        # reward agrégé (si fourni)
        if "reward" in outcome:
            r = clamp(float(outcome["reward"]), 0.0, 1.0)
            # EMA avec demi-vie douce
            self.ema_reward = round(0.7 * self.ema_reward + 0.3 * r, 4)

    # ----------- Sérialisation JSON pour Memory -----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "interaction_rule",
            "id": self.id,
            "version": self.version,
            "context_predicates": [asdict(p) for p in self.context_predicates],
            "tactic": {"name": self.tactic.name, "params": self.tactic.params},
            "effects": {k: {"alpha": v.alpha, "beta": v.beta, "decay": v.decay} for k, v in self.effects.items()},
            "created_ts": self.created_ts,
            "last_used_ts": self.last_used_ts,
            "usage_count": self.usage_count,
            "confidence": self.confidence,
            "ema_reward": self.ema_reward,
            "cooldown": self.cooldown,
            "provenance": self.provenance,
            "tags": list(self.tags),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InteractionRule":
        preds = [Predicate(**p) for p in d.get("context_predicates", [])]
        effs = {k: EffectPosterior(**v) for k, v in (d.get("effects") or {}).items()}
        t = d.get("tactic") or {}
        return InteractionRule(
            id=d["id"],
            version=d.get("version", "1.0"),
            context_predicates=preds,
            tactic=TacticSpec(name=t.get("name","noop"), params=t.get("params") or {}),
            effects=effs,
            created_ts=float(d.get("created_ts", _now())),
            last_used_ts=float(d.get("last_used_ts", 0.0)),
            usage_count=int(d.get("usage_count", 0)),
            confidence=float(d.get("confidence", 0.5)),
            ema_reward=float(d.get("ema_reward", 0.5)),
            cooldown=float(d.get("cooldown", 0.0)),
            provenance=d.get("provenance") or {},
            tags=list(d.get("tags") or ["social"])
        )

# ---------- Contexte de décision : builder depuis l’architecture ----------
class ContextBuilder:
    """
    Construit un contexte symbolique exploitable par les ``Predicate.test``.

    Ici on prépare un dictionnaire compact dédié au moteur de règles
    sociales (actes de dialogue, polarité, risques, persona…).  Il ne faut
    pas le confondre avec :class:`AGI_Evolutive.conversation.context.ContextBuilder`
    qui, lui, assemble un résumé narratif pour l'interface de conversation.
    """
    @staticmethod
    def build(arch, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            # actes de dialogue + topics mis par tes analyseurs
            "dialogue_act": getattr(getattr(arch, "conversation_state", None), "last_dialogue_act", None),
            "topic_cluster": getattr(getattr(arch, "conversation_state", None), "topic_cluster", None),
            # polarité / valence (heuristique locale)
            "polarity": getattr(getattr(arch, "affect_state", None), "polarity", 0.0),
            "valence": getattr(getattr(arch, "affect_state", None), "valence", 0.0),
            # sous-entendus/implicature (si ton analyzer en met une piste)
            "implicature_hint": getattr(getattr(arch, "conversation_state", None), "implicature_hint", None),
            # style utilisateur estimé / persona de l’agent
            "user_style": getattr(getattr(arch, "conversation_state", None), "user_style", {}),
            "persona_alignment": ContextBuilder._persona_alignment(arch),
            # risques / garde-fous policy
            "risk_level": ContextBuilder._risk_level(arch),
            # recence d’usage (pour anti-abus en sélection)
            "recence_usage": getattr(getattr(arch, "conversation_state", None), "recent_rule_usage", 0.0),
            # goal parent actif le plus saillant
            "parent_goal": ContextBuilder._active_parent_goal(arch),
        }
        if extra:
            ctx.update(extra)
        return ctx

    @staticmethod
    def _persona_alignment(arch) -> float:
        """
        Raccourci: retourne [0..1] sur "ce contexte correspond à mes valeurs/voix".
        Implémentation simple: si persona.values contient des concepts voisins du topic/act.
        (Tu peux enrichir via Ontology/Beliefs si dispo.)
        """
        try:
            persona = getattr(arch.self_model, "state", {}).get("persona", {})
            vals = set(v.lower() for v in persona.get("values", []))
            topics = set(getattr(getattr(arch, "conversation_state", None), "topic_cluster", []) or [])
            # intersection simple
            return clamp(len(vals & topics) / max(1, len(topics)))
        except Exception:
            return 0.5

    @staticmethod
    def _risk_level(arch) -> str:
        """
        Estimation simple: si Policy a récemment "needs_human/deny" dans ce fil → medium/high.
        """
        try:
            pol = getattr(arch, "policy", None)
            if pol and hasattr(pol, "recent_frictions"):
                fr = int(pol.recent_frictions(window_sec=600))
                if fr >= 2: return "high"
                if fr == 1: return "medium"
            return "low"
        except Exception:
            return "low"

    @staticmethod
    def _active_parent_goal(arch) -> Optional[str]:
        try:
            parents = getattr(arch.planner, "state", {}).get("parents", {})
            stack = getattr(arch.planner, "state", {}).get("active_stack", [])
            if not stack: return None
            g = stack[-1]
            return parents.get(g) or g
        except Exception:
            return None

# ---------- Aide: fabriquer rapidement une règle typique ----------
def make_rule_insinuation_banter(provenance: Optional[Dict[str, Any]] = None) -> InteractionRule:
    preds = [
        Predicate(key="dialogue_act", op="eq", value="insinuation", weight=1.2),
        Predicate(key="risk_level",  op="in", value=["low","medium"], weight=0.8),
        Predicate(key="persona_alignment", op="ge", value=0.3, weight=0.6),
    ]
    tactic = TacticSpec("banter_leger", {"soft": True, "max_len_delta": 40})
    return InteractionRule.build(preds, tactic, provenance=provenance or {"source":"curated"})
