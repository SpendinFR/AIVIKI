import os
import json
import time
import math
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import numpy as np

NEUTRAL = {"valence": 0.0, "arousal": 0.2, "dominance": 0.5}


@dataclass
class AffectState:
    t: float = field(default_factory=time.time)
    valence: float = 0.0   # -1 .. +1
    arousal: float = 0.2   # 0 .. 1
    dominance: float = 0.5 # 0 .. 1
    label: str = "neutral" # étiquette lisible


class EmotionEngine:
    """
    Moteur affectif persistant + modulateurs cognitifs.
    - Agrège les signaux (reward, erreurs, charge cognitive, succès)
    - Met à jour (valence, arousal, dominance) avec décroissance douce
    - Expose des modulateurs concrets (curiosity, priorities, tone, exploration)
    """

    def __init__(self,
                 path_state="data/mood.json",
                 path_dashboard="data/mood_dashboard.json"):
        os.makedirs(os.path.dirname(path_state), exist_ok=True)
        self.path_state = path_state
        self.path_dashboard = path_dashboard

        self.state = AffectState()
        self.last_save = 0.0
        self.last_modulators: Dict[str, Any] = {}
        self.bound = {
            "arch": None, "memory": None, "metacog": None, "goals": None, "language": None
        }
        self._load()

        # paramètres
        self.half_life_sec = 60.0 * 15.0  # retour progressif vers neutre (15 min)
        self.step_period = 2.0
        self._last_step = 0.0

        # poids d'appréciation (appraisal)
        self.w_error = -0.35
        self.w_success = +0.25
        self.w_reward = +0.30
        self.w_cog_load = +0.20   # hausse arousal
        self.w_fatigue = -0.20    # valence baisse

    # ---------- Binding ----------
    def bind(self, arch=None, memory=None, metacog=None, goals=None, language=None):
        self.bound.update({"arch": arch, "memory": memory, "metacog": metacog, "goals": goals, "language": language})

    # ---------- API externe ----------
    def register_event(self, kind: str, intensity: float = 0.4,
                       valence_hint: Optional[float] = None,
                       arousal_hint: Optional[float] = None,
                       dominance_hint: Optional[float] = None,
                       meta: Optional[Dict[str, Any]] = None):
        """
        Permet au reste du système d'injecter un événement émotionnel (facultatif).
        """
        dv = (valence_hint if valence_hint is not None else 0.0) * max(0.0, min(1.0, intensity))
        da = (arousal_hint if arousal_hint is not None else 0.0) * max(0.0, min(1.0, intensity))
        dd = (dominance_hint if dominance_hint is not None else 0.0) * 0.5 * max(0.0, min(1.0, intensity))
        self._nudge(dv, da, dd)

    def step(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_step < self.step_period):
            return
        self._last_step = now

        # 1) décroissance douce vers neutre
        self._decay_to_baseline(now)

        # 2) appraisal depuis métacognition & mémoire (si dispo)
        dv, da, dd = self._appraise_sources()

        # 3) nudge état
        self._nudge(dv, da, dd)

        # 4) modulateurs vers modules
        self.last_modulators = self._compute_modulators()

        # 5) dispatch doux (sans casser si APIs absentes)
        self._dispatch_modulators(self.last_modulators)

        # 6) persistance périodique
        if now - self.last_save > 10.0:
            self._save()

    def get_state(self) -> Dict[str, Any]:
        s = asdict(self.state)
        s["label"] = self._label()
        return s

    def get_modulators(self) -> Dict[str, Any]:
        return dict(self.last_modulators)

    # ---------- Internals ----------
    def _decay_to_baseline(self, now: float):
        """
        Exponentiel vers NEUTRAL avec demi-vie configurable.
        """

        def decay(old, target, half):
            dt = max(0.0, now - self.state.t)
            if half <= 0:
                return target
            factor = math.pow(0.5, dt / half)
            return target + (old - target) * factor

        self.state.valence = float(max(-1.0, min(1.0, decay(self.state.valence, NEUTRAL["valence"], self.half_life_sec))))
        self.state.arousal = float(max(0.0, min(1.0, decay(self.state.arousal, NEUTRAL["arousal"], self.half_life_sec))))
        self.state.dominance = float(max(0.0, min(1.0, decay(self.state.dominance, NEUTRAL["dominance"], self.half_life_sec))))
        self.state.t = now

    def _appraise_sources(self) -> (float, float, float):
        """
        Convertit signaux système -> deltas émotionnels.
        - erreurs récentes (metacog) → valence--
        - high cognitive load → arousal++
        - rewards/success dans memories → valence++
        - fatigue → valence--, dominance--
        """
        dv = da = dd = 0.0

        metacog = self.bound.get("metacog")
        arch = self.bound.get("arch")
        memory = self.bound.get("memory")

        # A) métacognition : erreurs / charge / fatigue
        try:
            if metacog and hasattr(metacog, "metacognitive_history"):
                evts = list(metacog.metacognitive_history.get("events", []))[-10:]
                for e in evts:
                    etype = getattr(e, "event_type", None) or (isinstance(e, dict) and e.get("event_type"))
                    signif = getattr(e, "significance", 0.0) if not isinstance(e, dict) else e.get("significance", 0.0)

                    if etype == "error_detected":
                        dv += self.w_error * signif
                        da += 0.1 * signif
                        dd -= 0.05 * signif
                    elif etype == "high_cognitive_load":
                        da += self.w_cog_load * max(0.2, e.get("cognitive_load", 0.5) if isinstance(e, dict) else 0.5)
                    elif etype == "performance_anomaly":
                        dv += -0.15 * signif
                        da += 0.05 * signif

            # fatigue (via resource_monitor assess_fatigue si dispo)
            if metacog and hasattr(metacog, "cognitive_monitoring"):
                rm = metacog.cognitive_monitoring.get("resource_monitor")
                if rm and hasattr(rm, "assess_fatigue"):
                    try:
                        f = rm.assess_fatigue(metacog.metacognitive_history, arch)
                        dv += self.w_fatigue * max(0.0, min(1.0, f))
                        dd -= 0.1 * max(0.0, min(1.0, f))
                    except Exception:
                        pass
        except Exception:
            pass

        # B) mémoire : rewards / succès / feedback
        try:
            recent = []
            if memory and hasattr(memory, "get_recent_memories"):
                recent = memory.get_recent_memories(n=40) or []
            elif memory and hasattr(memory, "memories"):
                recent = list(memory.memories)[-40:]

            for m in recent:
                # reward/valence implicite
                rew = m.get("reward")
                suc = m.get("success")
                emo = m.get("emotion")
                if isinstance(rew, (int, float)):
                    dv += self.w_reward * max(-1.0, min(1.0, float(rew)))
                    da += 0.05 * abs(float(rew))
                if suc is True:
                    dv += self.w_success * 0.6
                    dd += 0.05
                if isinstance(emo, dict):
                    dv += 0.15 * float(emo.get("valence", 0.0))
                    da += 0.10 * float(emo.get("arousal", 0.0))
        except Exception:
            pass

        # clamp doux
        return float(max(-0.5, min(0.5, dv))), float(max(-0.5, min(0.5, da))), float(max(-0.3, min(0.3, dd)))

    def _nudge(self, dv: float, da: float, dd: float):
        self.state.valence = float(max(-1.0, min(1.0, self.state.valence + dv)))
        self.state.arousal = float(max(0.0, min(1.0, self.state.arousal + da)))
        self.state.dominance = float(max(0.0, min(1.0, self.state.dominance + dd)))

    def _compute_modulators(self) -> Dict[str, Any]:
        """
        Traduit l'état émotionnel en modulations concrètes.
        """
        v = self.state.valence
        a = self.state.arousal
        d = self.state.dominance

        # curiosité boostée par arousal positif & valence neutre/+
        curiosity_gain = max(0.0, 0.2 * a + 0.1 * max(0.0, v))
        # exploration-exploitation
        exploration_rate = float(max(0.05, min(0.6, 0.15 + 0.5 * a - 0.2 * d)))
        # focus (arousal haut → tunnel)
        focus_narrowing = float(max(0.0, min(1.0, 0.3 + 0.5 * a)))

        # biais de priorité par domaine (exemples)
        goal_bias: Dict[str, float] = {}
        if v < -0.2:
            goal_bias.update({"résolution_problème": +0.15, "apprentissage": +0.05})
        if a > 0.6:
            goal_bias.update({"attention": +0.15, "prise_décision": +0.10})
        if v > 0.3:
            goal_bias.update({"social_cognition": +0.10, "langage": +0.05})

        # ton de langage (indices, pas d'imposition)
        tone = {
            "warmth": float(max(0.0, min(1.0, 0.5 + 0.4 * v))),
            "energy": float(max(0.0, min(1.0, 0.3 + 0.7 * a))),
            "assertiveness": float(max(0.0, min(1.0, 0.4 + 0.6 * (d - 0.5)))),
            "hedging": float(max(0.0, min(1.0, 0.6 - 0.4 * v)))
        }

        # activation globale (petit delta)
        activation_delta = float(max(-0.1, min(0.1, 0.05 * v + 0.05 * a - 0.03 * (0.5 - d))))

        return {
            "curiosity_gain": curiosity_gain,
            "exploration_rate": exploration_rate,
            "focus_narrowing": focus_narrowing,
            "goal_priority_bias": goal_bias,
            "language_tone": tone,
            "activation_delta": activation_delta,
            "label": self._label()
        }

    def _dispatch_modulators(self, mods: Dict[str, Any]):
        arch = self.bound.get("arch")
        goals = self.bound.get("goals")
        lang = self.bound.get("language")

        # activation globale
        try:
            if arch and hasattr(arch, "global_activation"):
                arch.global_activation = float(max(0.0, min(1.0, arch.global_activation + mods["activation_delta"])))
        except Exception:
            pass

        # biais de priorités d'objectifs (optionnel)
        try:
            if goals and hasattr(goals, "apply_emotional_bias"):
                goals.apply_emotional_bias(mods.get("goal_priority_bias", {}), mods.get("curiosity_gain", 0.0))
            elif arch is not None:
                # fallback : stocker pour que GoalSystem puisse le lire
                setattr(arch, "emotional_priority_bias", mods.get("goal_priority_bias", {}))
                setattr(arch, "emotional_curiosity_gain", mods.get("curiosity_gain", 0.0))
        except Exception:
            pass

        # ton de langage (optionnel)
        try:
            tone = mods.get("language_tone", {})
            if lang:
                if hasattr(lang, "set_style_hints"):
                    lang.set_style_hints(tone)
                else:
                    setattr(lang, "style_hints", tone)
        except Exception:
            pass

    def _label(self) -> str:
        v, a = self.state.valence, self.state.arousal
        if v > 0.35 and a > 0.6:
            return "joyful-energized"
        if v > 0.35 and a <= 0.6:
            return "content-calm"
        if v < -0.35 and a > 0.6:
            return "stressed-tense"
        if v < -0.35 and a <= 0.6:
            return "sad-calm"
        if abs(v) < 0.2 and a > 0.6:
            return "alert"
        return "neutral"

    # ---------- IO ----------
    def _save(self):
        payload = {"t": time.time(), "state": self.get_state(), "modulators": self.last_modulators}
        with open(self.path_state, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        dash = {
            "t": time.time(),
            "label": self._label(),
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "dominance": self.state.dominance,
            "mods": self.last_modulators
        }
        with open(self.path_dashboard, "w", encoding="utf-8") as f:
            json.dump(dash, f, ensure_ascii=False, indent=2)
        self.last_save = time.time()

    def _load(self):
        if not os.path.exists(self.path_state):
            return
        try:
            with open(self.path_state, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = data.get("state", {})
            self.state = AffectState(
                t=st.get("t", time.time()),
                valence=float(st.get("valence", 0.0)),
                arousal=float(st.get("arousal", 0.2)),
                dominance=float(st.get("dominance", 0.5)),
                label=st.get("label", "neutral")
            )
        except Exception:
            self.state = AffectState()
