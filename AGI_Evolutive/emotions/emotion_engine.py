"""
EmotionEngineV2 — moteur émotionnel évolutif, compatible avec l'existant
-------------------------------------------------------------------------
- PAD sur 2 échelles : épisode (rapide) + humeur (lente, set‑point dynamique)
- Appraisal *pluginisé* (erreur, succès, récompense, charge cog., fatigue, social)
  avec poids apprenants en ligne (SGD borné) et confiance par signal
- Incertitude estimée et propagée → exploration, focus, ton, safety gate
- Journalisation JSONL des épisodes + dashboard JSON d'état (poids, mood, etc.)
- **Compat 100%** avec l'existant :
  - API: bind(), register_event(), step(), get_modulators(), get_state(),
          update_from_recent_memories(), modulate_homeostasis()
  - Modulateurs: mêmes clés + alias 
    *tone* **et** *language_tone*, *goal_priority_bias* **dict** + *goal_priority_bias_scalar*
  - Bump explicite de arch.global_activation via activation_delta
  - Cible de décroissance configurable: vers "mood" (par défaut) ou "neutral"

Auteur: Toi (refonte assistée) — 2025-10-19
Licence: MIT
"""
from __future__ import annotations

import os
import json
import time
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple

# ========================= Utilitaires ========================= #

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def json_sanitize(obj: Any) -> Any:
    """Sanitise en JSON sans dépendances externes."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_sanitize(v) for v in obj]
        try:
            if obj != obj:  # NaN
                return None
        except Exception:
            pass
        return str(obj)


NEUTRAL = {"valence": 0.0, "arousal": 0.2, "dominance": 0.5}

# ========================= États ========================= #

@dataclass
class AffectState:
    t: float = field(default_factory=time.time)
    valence: float = NEUTRAL["valence"]      # -1..1
    arousal: float = NEUTRAL["arousal"]      #  0..1
    dominance: float = NEUTRAL["dominance"]  #  0..1
    label: str = "neutral"


@dataclass
class MoodState:
    t: float = field(default_factory=time.time)
    valence: float = NEUTRAL["valence"]
    arousal: float = 0.15
    dominance: float = NEUTRAL["dominance"]


@dataclass
class EmotionEpisode:
    id: str
    onset: float
    dt: float
    dv: float
    da: float
    dd: float
    label: str
    causes: List[Tuple[str, float]]  # (plugin, contribution)
    confidence: float
    meta: Dict[str, Any]


# ========================= Appraisal (plugins) ========================= #

@dataclass
class AppraisalOutput:
    dv: float
    da: float
    dd: float
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class AppraisalPlugin:
    name: str = "base"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        raise NotImplementedError


class CognitiveLoadPlugin(AppraisalPlugin):
    name = "cog_load"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        load = float(ctx.get("cognitive_load", 0.5) or 0.0)
        conf = 0.7 if "cognitive_load" in ctx else 0.3
        return AppraisalOutput(dv=0.0, da=0.30 * load, dd=0.0, confidence=conf, meta={"load": load})


class ErrorPlugin(AppraisalPlugin):
    name = "error"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        errs = ctx.get("recent_errors", []) or []
        n = float(len(errs))
        magnitude = clip(n / 5.0, 0.0, 1.0)
        conf = 0.6 if n > 0 else 0.25
        return AppraisalOutput(dv=-0.40 * magnitude, da=0.10 * magnitude, dd=-0.20 * magnitude,
                               confidence=conf, meta={"n_errors": n})


class SuccessPlugin(AppraisalPlugin):
    name = "success"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        succ = float(ctx.get("recent_success", 0.0) or 0.0)  # ∈[0,1]
        conf = 0.6 if "recent_success" in ctx else 0.3
        return AppraisalOutput(dv=+0.30 * succ, da=+0.05 * succ, dd=+0.20 * succ,
                               confidence=conf, meta={"success": succ})


class RewardPlugin(AppraisalPlugin):
    name = "reward"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        rew = float(ctx.get("reward_signal", 0.0) or 0.0)
        rew = clip(rew, -1.0, 1.0)
        conf = 0.7 if "reward_signal" in ctx else 0.3
        return AppraisalOutput(dv=0.35 * rew, da=0.10 * abs(rew), dd=0.05 * rew,
                               confidence=conf, meta={"reward": rew})


class FatiguePlugin(AppraisalPlugin):
    name = "fatigue"
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        fatigue = float(ctx.get("fatigue", 0.0) or 0.0)  # 0–1
        conf = 0.6 if "fatigue" in ctx else 0.25
        return AppraisalOutput(dv=-0.20 * fatigue, da=-0.15 * fatigue, dd=-0.05 * fatigue,
                               confidence=conf, meta={"fatigue": fatigue})


class SocialFeedbackPlugin(AppraisalPlugin):
    name = "social"
    POS = ("bravo", "merci", "thanks", "bien", "good", "+1")
    NEG = ("mauvais", "nul", "pas bien", "bad", "-1", "wrong")
    def __call__(self, ctx: Dict[str, Any]) -> AppraisalOutput:
        cues = ctx.get("social_cues", []) or []
        pos = sum(any(p in str(c).lower() for p in self.POS) for c in cues)
        neg = sum(any(n in str(c).lower() for n in self.NEG) for c in cues)
        score = clip((pos - neg) / 3.0, -1.0, 1.0)
        conf = 0.5 if cues else 0.2
        return AppraisalOutput(dv=0.25 * score, da=0.05 * abs(score), dd=0.05 * score,
                               confidence=conf, meta={"pos": pos, "neg": neg})


class AppraisalAggregator:
    def __init__(self, plugins: List[AppraisalPlugin], lr: float = 0.02, w_max: float = 3.0):
        self.plugins: Dict[str, AppraisalPlugin] = {p.name: p for p in plugins}
        self.w: Dict[str, float] = {p.name: 1.0 for p in plugins}
        self.lr = float(lr)
        self.w_max = float(w_max)

    def step(self, ctx: Dict[str, Any], quality: Optional[float]) -> Tuple[float, float, float, Dict[str, float]]:
        dv = da = dd = 0.0
        parts: Dict[str, float] = {}
        cache: Dict[str, AppraisalOutput] = {}
        for name, p in self.plugins.items():
            out = p(ctx)
            cache[name] = out
            contrib = float(self.w[name]) * float(out.confidence)
            dv += contrib * out.dv
            da += contrib * out.da
            dd += contrib * out.dd
            parts[name] = contrib * (abs(out.dv) + 0.5 * abs(out.da) + 0.3 * abs(out.dd))
        if quality is not None:
            target = clip(quality, -1.0, 1.0)
            for name, out in cache.items():
                signal = out.dv + 0.5 * out.da + 0.3 * out.dd
                grad = target * signal * out.confidence
                self.w[name] = clip(self.w[name] + self.lr * grad, 0.0, self.w_max)
        return dv, da, dd, parts


# ========================= Labelling ========================= #

EMO_LABELS = [
    ("elated",       (0.6, 0.6, 0.5)),
    ("excited",      (0.5, 0.7, 0.5)),
    ("content",      (0.4, 0.3, 0.5)),
    ("calm",         (0.2, 0.2, 0.5)),
    ("neutral",      (0.0, 0.2, 0.5)),
    ("tense",        (-0.2, 0.6, 0.4)),
    ("frustrated",   (-0.5, 0.6, 0.4)),
    ("sad",          (-0.5, 0.3, 0.4)),
    ("bored",        (-0.2, 0.1, 0.5)),
]


def label_from_pad(v: float, a: float, d: float) -> str:
    best, bd = "neutral", 9e9
    for name, (lv, la, ld) in EMO_LABELS:
        dist = (v - lv) ** 2 + (a - la) ** 2 + 0.2 * (d - ld) ** 2
        if dist < bd: best, bd = name, dist
    return best


# ========================= Moteur principal ========================= #

class EmotionEngine:
    """Moteur émotionnel évolutif — **compat 100%** avec l'existant.

    API stable: bind(), register_event(), step(), get_modulators(), get_state(),
                update_from_recent_memories(), modulate_homeostasis().
    """
    def __init__(self,
                 path_state: str = "data/mood.json",
                 path_dashboard: str = "data/mood_dashboard.json",
                 path_log: str = "data/mood_episodes.jsonl",
                 half_life_sec: float = 60.0 * 15.0,
                 step_period: float = 2.0,
                 mood_half_life_sec: float = 60.0 * 60.0 * 6.0,  # 6h
                 decay_target: str = "mood",  # "mood" (défaut) ou "neutral"
                 seed: Optional[int] = None):
        os.makedirs(os.path.dirname(path_state) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(path_dashboard) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(path_log) or ".", exist_ok=True)
        self.path_state = path_state
        self.path_dashboard = path_dashboard
        self.path_log = path_log

        self.state = AffectState()
        self.mood = MoodState()
        self._last_step = 0.0
        self.half_life_sec = float(half_life_sec)
        self.step_period = float(step_period)
        self.mood_half_life_sec = float(mood_half_life_sec)
        self.decay_target = decay_target
        self.rng = (seed and int(seed)) or None

        # Plugins & agrégateur
        self.aggregator = AppraisalAggregator([
            CognitiveLoadPlugin(), ErrorPlugin(), SuccessPlugin(), RewardPlugin(), FatiguePlugin(), SocialFeedbackPlugin()
        ])

        # Liaison vers d'autres modules
        self.bound: Dict[str, Any] = {"arch": None, "memory": None, "metacog": None, "goals": None, "language": None}

        # Épisodes récents
        self._recent_episodes: List[EmotionEpisode] = []
        self.max_recent_episodes = 200

        # Cache modulators
        self.last_modulators: Dict[str, Any] = {}

        # Charger si présent
        self.load()

    # ---------- Binding ----------
    def bind(self, arch=None, memory=None, metacog=None, goals=None, language=None):
        self.bound.update({"arch": arch, "memory": memory, "metacog": metacog, "goals": goals, "language": language})
        return self

    # ---------- API externe ----------
    def register_event(self, kind: str, intensity: float = 0.4,
                       valence_hint: Optional[float] = None,
                       arousal_hint: Optional[float] = None,
                       dominance_hint: Optional[float] = None,
                       confidence: float = 1.0,
                       meta: Optional[Dict[str, Any]] = None):
        m = float(clip(intensity, 0.0, 1.0))
        dv = (valence_hint if valence_hint is not None else 0.0) * m
        da = (arousal_hint if arousal_hint is not None else 0.0) * m
        dd = (dominance_hint if dominance_hint is not None else 0.0) * 0.5 * m
        self._nudge(dv, da, dd, source=f"event:{kind}", confidence=confidence, meta=meta or {})

    def get_modulators(self) -> Dict[str, Any]:
        self.step(force=True)
        if not self.last_modulators:
            self.last_modulators = self._compute_modulators()
        return dict(self.last_modulators)

    def get_state(self) -> Dict[str, Any]:
        return {
            "episode": asdict(self.state),
            "mood": asdict(self.mood),
            "uncertainty": self._estimate_uncertainty(),
            "label": self.state.label,
            "weights": dict(self.aggregator.w),
            "recent_causes": [
                {"id": e.id, "label": e.label, "causes": e.causes[-3:], "dv": e.dv, "da": e.da, "dd": e.dd, "confidence": e.confidence, "dt": e.dt}
                for e in self._recent_episodes[-5:]
            ],
        }

    # ---------- Compat: helpers hérités ----------
    def update_from_recent_memories(self, recent: List[Dict[str, Any]]):
        if not isinstance(recent, list):
            recent = []
        positive = ("bravo", "merci", "good", "bien")
        negative = ("erreur", "fail", "mauvais", "wrong", "error")
        for m in recent:
            text = str((m or {}).get("text", "")).lower()
            kind = str((m or {}).get("kind", "")).lower()
            if any(t in text for t in positive):
                self.register_event("positive_feedback", 0.5, valence_hint=+0.7, arousal_hint=+0.2, dominance_hint=+0.1)
            if "error" in kind or any(t in text for t in negative):
                self.register_event("error_feedback", 0.6, valence_hint=-0.7, arousal_hint=+0.3, dominance_hint=-0.2)
            if "?" in text:
                self.register_event("curiosity_signal", 0.3, valence_hint=+0.1, arousal_hint=+0.2)
        self.step(force=True)

    def modulate_homeostasis(self, homeostasis) -> None:
        if homeostasis is None:
            return
        self.step(force=True)
        mods = self.last_modulators or self._compute_modulators()
        curiosity_boost = float(mods.get("curiosity_gain", 0.0))
        activation_delta = float(mods.get("activation_delta", 0.0))
        def _current(name: str) -> float:
            try:
                return float(homeostasis.state.get("drives", {}).get(name, 0.5))
            except Exception:
                return 0.5
        def _apply(name: str, delta: float) -> None:
            try:
                if hasattr(homeostasis, "adjust_drive"):
                    homeostasis.adjust_drive(name, delta)
                    return
            except Exception:
                pass
            # fallback direct
            drives_dict = homeostasis.state.setdefault("drives", {})
            drives_dict[name] = clip(_current(name) + delta, 0.0, 1.0)
            if hasattr(homeostasis, "_save"):
                try: homeostasis._save()
                except Exception: pass
        _apply("curiosity", +0.05 + 0.10 * curiosity_boost)
        _apply("task_activation", activation_delta)

    # ---------- Tick principal ----------
    def step(self, force: bool = False, quality: Optional[float] = None, now: Optional[float] = None):
        now = float(now or time.time())
        if not force and (now - self._last_step) < self.step_period:
            return

        # 1) Décroissance épisode → baseline (mood ou neutral selon flag)
        self._decay_episode(now)

        # 2) Contexte + appraisal via agrégateur (avec proxy qualité)
        ctx = self._collect_context()
        dv, da, dd, parts = self.aggregator.step(ctx, quality=quality if quality is not None else self._proxy_quality(ctx))

        # 3) Appliquer delta + journaliser
        self._nudge(dv, da, dd, source="aggregator", confidence=self._confidence_from_ctx(ctx),
                    meta={"parts": parts, "ctx_keys": list(ctx.keys())})

        # 4) Mise à jour humeur (filtre lent)
        self._update_mood(now)

        # 5) Recalcul modulateurs & dispatch
        self.last_modulators = self._compute_modulators()
        self._dispatch_modulators(self.last_modulators)

        # 6) Persistance légère
        self.save()
        self._last_step = now

    # ========================= Interne ========================= #
    def _collect_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        metacog = self.bound.get("metacog")
        memory = self.bound.get("memory")
        language = self.bound.get("language")
        arch = self.bound.get("arch")
        if metacog is not None:
            ctx.setdefault("cognitive_load", getattr(metacog, "load", None))
            # fatigue (si disponible)
            try:
                mon = getattr(getattr(metacog, "cognitive_monitoring", {}), "resource_monitor", None)
                if mon and hasattr(mon, "assess_fatigue"):
                    ctx.setdefault("fatigue", float(clip(mon.assess_fatigue(getattr(metacog, "metacognitive_history", None), arch), 0.0, 1.0)))
            except Exception:
                pass
            ctx.setdefault("recent_errors", getattr(metacog, "recent_errors", None))
            ctx.setdefault("reward_signal", getattr(metacog, "reward_signal", None))
            ctx.setdefault("recent_success", getattr(metacog, "recent_success", None))
        if memory is not None and hasattr(memory, "get_recent_memories"):
            try: ctx.setdefault("recent_memories", memory.get_recent_memories(n=40))
            except Exception: pass
        if language is not None:
            ctx.setdefault("social_cues", getattr(language, "recent_user_cues", None))
        ctx.setdefault("time_of_day", time.localtime().tm_hour)
        return {k: v for k, v in ctx.items() if v is not None}

    def _confidence_from_ctx(self, ctx: Dict[str, Any]) -> float:
        filled = sum(1 for _ in ctx.items())
        return clip(0.2 + 0.05 * filled, 0.2, 1.0)

    def _proxy_quality(self, ctx: Dict[str, Any]) -> Optional[float]:
        succ = float(ctx.get("recent_success", 0.0) or 0.0)
        errs = float(len(ctx.get("recent_errors", []) or []))
        fatigue = float(ctx.get("fatigue", 0.0) or 0.0)
        return clip(succ - 0.2 * errs - 0.3 * fatigue, -1.0, 1.0)

    def _decay_episode(self, now: float):
        dt = max(0.001, now - self.state.t)
        hl = max(1e-3, self.half_life_sec)
        k = 0.6931 / hl
        def towards(curr, base):
            return base + (curr - base) * math.exp(-k * dt)
        if self.decay_target == "neutral":
            base_v, base_a, base_d = 0.0, 0.2, 0.5
        else:  # mood
            base_v, base_a, base_d = self.mood.valence, max(0.05, self.mood.arousal), self.mood.dominance
        self.state.valence = clip(towards(self.state.valence, base_v), -1.0, 1.0)
        self.state.arousal = clip(towards(self.state.arousal, base_a), 0.0, 1.0)
        self.state.dominance = clip(towards(self.state.dominance, base_d), 0.0, 1.0)
        self.state.t = now
        self.state.label = label_from_pad(self.state.valence, self.state.arousal, self.state.dominance)

    def _update_mood(self, now: float):
        dt = max(0.001, now - self.mood.t)
        hl = max(1e-3, self.mood_half_life_sec)
        k = 0.6931 / hl
        def lowpass(m, s):
            return m + (s - m) * (1.0 - math.exp(-k * dt))
        self.mood.valence = clip(lowpass(self.mood.valence, self.state.valence), -1.0, 1.0)
        self.mood.arousal = clip(lowpass(self.mood.arousal, self.state.arousal * 0.8), 0.05, 1.0)
        self.mood.dominance = clip(lowpass(self.mood.dominance, self.state.dominance), 0.0, 1.0)
        self.mood.t = now

    def _estimate_uncertainty(self) -> float:
        base = clip(0.6 * self.state.arousal + 0.4 * (1.0 - self.state.dominance), 0.0, 1.0)
        return clip(0.5 * base + 0.5 * (0.5 - 0.5 * self.mood.dominance), 0.0, 1.0)

    def _nudge(self, dv: float, da: float, dd: float, source: str, confidence: float, meta: Dict[str, Any]):
        self.state.valence = clip(self.state.valence + dv, -1.0, 1.0)
        self.state.arousal = clip(self.state.arousal + da, 0.0, 1.0)
        self.state.dominance = clip(self.state.dominance + dd, 0.0, 1.0)
        self.state.label = label_from_pad(self.state.valence, self.state.arousal, self.state.dominance)
        ep = EmotionEpisode(
            id=str(uuid.uuid4()), onset=time.time(), dt=self.step_period,
            dv=dv, da=da, dd=dd, label=self.state.label, confidence=confidence,
            causes=sorted([(k, float(v)) for k, v in (meta.get("parts") or {}).items()], key=lambda x: -x[1])[:5],
            meta={k: v for k, v in meta.items() if k != "parts"}
        )
        self._recent_episodes.append(ep)
        if len(self._recent_episodes) > self.max_recent_episodes:
            self._recent_episodes = self._recent_episodes[-self.max_recent_episodes:]
        try:
            with open(self.path_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_sanitize(asdict(ep)), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _goal_priority_bias_dict(self, v: float, a: float, d: float) -> Dict[str, float]:
        """Reconstruit un dict de biais par domaines (compat ancien code)."""
        bias: Dict[str, float] = {}
        # heuristiques simples mais stables
        if v < -0.2:
            bias["résolution_problème"] = bias.get("résolution_problème", 0.0) + 0.15
            bias["apprentissage"] = bias.get("apprentissage", 0.0) + 0.05
        if a > 0.6:
            bias["attention"] = bias.get("attention", 0.0) + 0.15
            bias["prise_décision"] = bias.get("prise_décision", 0.0) + 0.10
        if v > 0.3:
            bias["social_cognition"] = bias.get("social_cognition", 0.0) + 0.10
            bias["langage"] = bias.get("langage", 0.0) + 0.05
        # clé globale pour consommateurs génériques
        bias["global"] = clip(0.15 * v + 0.10 * (d - 0.5), -0.3, 0.3)
        return bias

    def _compute_modulators(self) -> Dict[str, Any]:
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance
        unc = self._estimate_uncertainty()

        curiosity_gain = clip(0.2 + 0.4 * max(0.0, v) + 0.2 * (1.0 - unc), 0.0, 1.0)
        exploration_rate = clip(0.15 + 0.5 * unc + 0.2 * a, 0.0, 1.0)
        focus_narrowing = clip(0.3 + 0.5 * a - 0.3 * unc, 0.0, 1.0)

        tone = {
            "warmth": clip(0.55 + 0.35 * v - 0.15 * unc, 0.0, 1.0),
            "energy": clip(0.30 + 0.60 * a, 0.0, 1.0),
            "assertiveness": clip(0.30 + 0.50 * d - 0.25 * unc, 0.0, 1.0),
            "hedging": clip(0.25 + 0.50 * unc - 0.15 * d, 0.0, 1.0),
        }
        # Safety tone gate
        if d < 0.4 and unc > 0.5:
            tone["assertiveness"] = min(tone["assertiveness"], 0.45)

        activation_delta = clip(0.25 * a + 0.15 * (v - 0.2), -0.3, 0.7)
        goal_priority_bias_scalar = clip(0.15 * v + 0.10 * (d - 0.5), -0.3, 0.3)
        goal_priority_bias = self._goal_priority_bias_dict(v, a, d)

        mods = {
            "curiosity_gain": curiosity_gain,
            "exploration_rate": exploration_rate,
            "focus_narrowing": focus_narrowing,
            "tone": tone,
            "language_tone": dict(tone),  # alias compat
            "activation_delta": activation_delta,
            "goal_priority_bias": goal_priority_bias,  # dict (compat)
            "goal_priority_bias_scalar": goal_priority_bias_scalar,  # scalaire (nouveau)
            "uncertainty": unc,
            "label": self.state.label,  # alias compat si l'ancien code lisait label ici
        }
        return mods

    def _dispatch_modulators(self, mods: Dict[str, Any]):
        arch = self.bound.get("arch")
        lang = self.bound.get("language")
        goals = self.bound.get("goals")

        def call(obj, name: str, *args, **kwargs):
            try:
                fn = getattr(obj, name, None)
                if callable(fn): fn(*args, **kwargs)
            except Exception:
                pass

        # Hook générique
        if arch is not None:
            call(arch, "on_affect_modulators", mods)

        # **Compat 1**: Ton — set_style_hints (ancien) OU set_tone_param (nouveau)
        tone = mods.get("language_tone") or mods.get("tone") or {}
        if lang is not None:
            if hasattr(lang, "set_style_hints"):
                try: lang.set_style_hints(tone)
                except Exception: pass
            elif hasattr(lang, "set_tone_param"):
                for k, v in tone.items():
                    try: lang.set_tone_param(k, v)
                    except Exception: pass

        # **Compat 2**: Goal priority — dict (apply_emotional_bias) OU scalaire (set_priority_bias)
        if goals is not None:
            if hasattr(goals, "apply_emotional_bias"):
                try: goals.apply_emotional_bias(mods.get("goal_priority_bias", {}), mods.get("curiosity_gain", 0.0))
                except Exception: pass
            elif hasattr(goals, "set_priority_bias"):
                try: goals.set_priority_bias(float(mods.get("goal_priority_bias_scalar", 0.0)))
                except Exception: pass

        # **Compat 3**: Global activation bump explicite
        if arch is not None and hasattr(arch, "global_activation"):
            try:
                ga = float(getattr(arch, "global_activation", 0.5))
                ga = clip(ga + float(mods.get("activation_delta", 0.0)), 0.0, 1.0)
                setattr(arch, "global_activation", ga)
            except Exception:
                pass

    # ========================= Persistance ========================= #
    def save(self):
        payload = {
            "state": asdict(self.state),
            "mood": asdict(self.mood),
            "weights": dict(self.aggregator.w),
            "last_modulators": self.last_modulators,
            "t": time.time(),
        }
        try:
            with open(self.path_state, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(payload), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        dash = {
            "t": time.time(),
            "label": self.state.label,
            "v": round(self.state.valence, 3),
            "a": round(self.state.arousal, 3),
            "d": round(self.state.dominance, 3),
            "m_v": round(self.mood.valence, 3),
            "m_a": round(self.mood.arousal, 3),
            "m_d": round(self.mood.dominance, 3),
            "unc": round(self._estimate_uncertainty(), 3),
            "weights": dict(self.aggregator.w),
        }
        try:
            with open(self.path_dashboard, "w", encoding="utf-8") as f:
                json.dump(json_sanitize(dash), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load(self):
        try:
            if os.path.exists(self.path_state):
                with open(self.path_state, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st = data.get("state", {}) or data.get("episode", {}) or {}
                md = data.get("mood", {})
                self.state = AffectState(
                    t=float(st.get("t", time.time())),
                    valence=float(st.get("valence", NEUTRAL["valence"])),
                    arousal=float(st.get("arousal", NEUTRAL["arousal"])),
                    dominance=float(st.get("dominance", NEUTRAL["dominance"])),
                    label=str(st.get("label", "neutral")),
                )
                self.mood = MoodState(
                    t=float(md.get("t", time.time())),
                    valence=float(md.get("valence", NEUTRAL["valence"])),
                    arousal=float(md.get("arousal", 0.15)),
                    dominance=float(md.get("dominance", NEUTRAL["dominance"])),
                )
                w = data.get("weights", {}) or data.get("state", {}).get("weights", {})
                for k, v in w.items():
                    if k in self.aggregator.w:
                        self.aggregator.w[k] = float(v)
        except Exception:
            # fallback silencieux
            self.state = AffectState()
            self.mood = MoodState()


# ========================= Cli léger ========================= #
if __name__ == "__main__":
    eng = EmotionEngine()
    # Exemple de bind minimal tolérant
    class _Meta: load=0.6; recent_errors=[1,2]; recent_success=0.3; reward_signal=0.1
    eng.bind(metacog=_Meta())
    for _ in range(5):
        eng.step(force=True)
        print(eng.get_modulators())
