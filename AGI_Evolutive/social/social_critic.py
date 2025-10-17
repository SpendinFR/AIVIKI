# Social Critic : calcule des résultats observables + un reward multi-source
# et met à jour les InteractionRule (postérieurs + ema_reward).
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import time, math, re, json

from AGI_Evolutive.social.adaptive_lexicon import AdaptiveLexicon
from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, ContextBuilder, clamp
)


def _now() -> float: return time.time()

# ----------------- lexiques FR simples (tu peux enrichir) -----------------
POS_MARKERS = [
    "j'adore", "génial", "top", "parfait", "bravo", "merci", "super", "nickel",
    "impeccable", "parfaitement", "très bien", "bien vu", "ça me va", "ok c'est bon"
]
NEG_MARKERS = [
    "trop", "arrête", "stop", "pas ça", "relou", "ça me saoule", "mauvais", "bof",
    "je n'aime pas", "non pas comme ça", "pas terrible", "déçu", "insupportable"
]
ACCEPTANCE = ["ok", "d'accord", "merci", "compris", "noté", "ça marche", "bien reçu"]


def _contains_any(s: str, words: List[str]) -> bool:
    s = s.lower()
    return any(w in s for w in words)


def _sentiment_heuristic(s: str) -> float:
    """Renvoie une valence approx ∈ [-1,1]. Zéro si neutre."""
    s = (s or "").lower()
    pos = sum(1 for w in POS_MARKERS if w in s)
    neg = sum(1 for w in NEG_MARKERS if w in s)
    if pos == 0 and neg == 0:
        return 0.0
    val = (pos - neg) / max(1.0, (pos + neg))
    return max(-1.0, min(1.0, val))


def _acceptance_marker(s: str) -> bool:
    return _contains_any(s, ACCEPTANCE)


# ----------------- Critic -----------------
class SocialCritic:
    """
    Calcule un outcome post-réponse + reward multi-source avec incertitude.
    Sources (exemples, toutes optionnelles) :
      - Explicite (markers FR positifs/négatifs)
      - Implicite : baisse des pending_questions, poursuite du fil, valence heuristique
      - Policy friction (pénalité si friction récente)
      - Consistance identitaire (bonus si persona_alignment acceptable)
    Poids configurables via data/social_critic_config.json (optionnel).
    """

    def __init__(self, arch):
        self.arch = arch
        self.cfg = self._load_cfg()
        self.lex = getattr(self.arch, "lexicon", None)
        needs_adaptive = self.lex is None or not all(
            callable(getattr(self.lex, attr, None)) for attr in ("match", "observe_message")
        )
        if needs_adaptive:
            # seed avec tes POS/NEG statiques pour ne pas “perdre” ton actuel
            self.arch.lexicon_seeds = {
                "pos": [
                    "j'adore",
                    "génial",
                    "top",
                    "parfait",
                    "bravo",
                    "merci",
                    "super",
                    "nickel",
                    "impeccable",
                    "parfaitement",
                    "très bien",
                    "bien vu",
                    "ça me va",
                    "ok c'est bon",
                ],
                "neg": [
                    "trop",
                    "arrête",
                    "stop",
                    "pas ça",
                    "relou",
                    "ça me saoule",
                    "mauvais",
                    "bof",
                    "je n'aime pas",
                    "non pas comme ça",
                    "pas terrible",
                    "déçu",
                    "insupportable",
                ],
            }
            self.lex = AdaptiveLexicon(self.arch)
            setattr(self.arch, "lexicon", self.lex)

        self._reward_history: List[float] = []
        self._reward_perf_baseline: float = 0.6
        self._last_calibration_ts: float = 0.0

    def _load_cfg(self) -> Dict[str, Any]:
        path = getattr(self.arch, "social_critic_cfg_path", "data/social_critic_config.json")
        try:
            import os
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        # défaut raisonnable
        return {
            "weights": {
                "explicit_feedback": 0.45,   # j'adore / trop / etc.
                "uncertainty_delta": 0.20,   # baisse des pending_questions
                "continue_dialogue": 0.10,   # fil continue
                "valence":           0.15,   # valence heuristique
                "acceptance":        0.10,   # ok / d'accord / merci
                "identity_consist":  0.05,   # alignment persona
                "policy_friction":  -0.15    # pénalité si friction
            },
            "min_confidence": 0.15
        }

    # ----------- contexte auxiliaire avant/après ----------
    def _pending_questions_count(self) -> int:
        try:
            qm = getattr(self.arch, "question_manager", None)
            return len(getattr(qm, "pending_questions", []) or [])
        except Exception:
            return 0

    def _policy_friction_recent(self, window_sec: int = 300) -> int:
        try:
            pol = getattr(self.arch, "policy", None)
            if pol and hasattr(pol, "recent_frictions"):
                return int(pol.recent_frictions(window_sec=window_sec))
        except Exception:
            pass
        return 0

    # ----------- calcul de l'outcome ----------
    def compute_outcome(self,
                        user_msg: str,
                        decision_trace: Dict[str, Any],
                        pre_ctx: Optional[Dict[str, Any]] = None,
                        post_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retourne: {
          "reduced_uncertainty": bool,
          "continued": bool,
          "valence": float [-1..1],
          "accepted": bool,
          "reward": float [0..1],
          "confidence": float [0..1],
          "components": { ... contributions ... }
        }
        """
        W = self.cfg["weights"]

        # pré / post comptage des incertitudes
        pre_q  = int((pre_ctx or {}).get("pending_questions_count", self._pending_questions_count()))
        post_q = int((post_ctx or {}).get("pending_questions_count", self._pending_questions_count()))
        reduced_unc = (post_q < pre_q)

        # fil continu : trivial si user a répondu et n'a pas “coupé” grossièrement
        cont = True
        low = (user_msg or "").strip().lower()
        if not low or low in {"stop", "bye", "au revoir"}:
            cont = False

        # valence + acceptance
        val = _sentiment_heuristic(user_msg or "")
        acc = _acceptance_marker(user_msg or "")

        # feedback explicite (hybride: statique + lexique appris)
        user_id = getattr(self.arch, "user_id", None)
        match_fn = getattr(self.lex, "match", None)
        exp_pos = _contains_any(low, POS_MARKERS) or (
            callable(match_fn) and match_fn(user_msg, polarity="pos", user_id=user_id)
        )
        exp_neg = _contains_any(low, NEG_MARKERS) or (
            callable(match_fn) and match_fn(user_msg, polarity="neg", user_id=user_id)
        )
        explicit = 0.5
        if exp_pos and not exp_neg:
            explicit = 1.0
        elif exp_neg and not exp_pos:
            explicit = 0.0
        elif exp_pos and exp_neg:
            explicit = 0.5

        # identité (alignement persona du post-contexte si dispo)
        identity_consist = float((post_ctx or {}).get("persona_alignment", 0.5))

        # policy friction
        frictions = self._policy_friction_recent(window_sec=300)
        pol_pen = 1.0 if frictions == 0 else (0.7 if frictions == 1 else 0.4)

        # agrégation reward (0..1), pondérée
        # mapping valence [-1..1] -> [0..1]
        val01 = (val + 1.0) / 2.0
        reward = 0.0
        comp = {}

        parts = [
            ("explicit_feedback", explicit),
            ("uncertainty_delta", 1.0 if reduced_unc else 0.0),
            ("continue_dialogue", 1.0 if cont else 0.0),
            ("valence",           val01),
            ("acceptance",        1.0 if acc else 0.0),
            ("identity_consist",  identity_consist),
        ]
        for name, v in parts:
            w = float(W.get(name, 0.0))
            reward += w * v
            comp[name] = {"w": w, "v": round(v, 3), "contrib": round(w * v, 3)}

        # policy friction (pénalité multiplicative)
        reward = reward * max(0.0, min(1.0, 1.0 + float(W.get("policy_friction", -0.15)) * (1.0 - pol_pen)))
        reward = clamp(reward, 0.0, 1.0)

        # confiance de l'estimation (plus on a de signaux forts, plus c'est fiable)
        support = 0.0
        support += 0.35 if (exp_pos or exp_neg) else 0.0
        support += 0.20 if reduced_unc else 0.0
        support += 0.10 if cont else 0.0
        support += 0.10 if acc else 0.0
        support += 0.10  # base
        confidence = clamp(max(self.cfg.get("min_confidence", 0.15), support), 0.0, 1.0)

        # apprentissage du lexique (pas binaire — reward et confidence pondèrent)
        observe_fn = getattr(self.lex, "observe_message", None)
        if callable(observe_fn):
            try:
                observe_fn(
                    user_msg,
                    reward01=reward,
                    confidence=confidence,
                    user_id=user_id,
                )
            except Exception:
                pass

        return {
            "reduced_uncertainty": reduced_unc,
            "continued": cont,
            "valence": round(val, 3),
            "accepted": acc,
            "reward": round(reward, 4),
            "confidence": round(confidence, 3),
            "components": comp
        }

    # ----------- application à la règle ----------
    def update_rule_with_outcome(self, rule_id: str, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Charge la règle depuis la mémoire, observe l'outcome et persiste.
        Retourne le dict final persisté (ou None si pas trouvé).
        """
        # récupérer la règle la plus récente (kind="interaction_rule", id=rule_id)
        try:
            rules = self.arch.memory.get_recent_memories(kind="interaction_rule", limit=500) or []
        except Exception:
            rules = []
        rule_dict = None
        for r in rules:
            if r.get("id") == rule_id:
                rule_dict = r
                break
        if not rule_dict:
            return None

        try:
            rule = InteractionRule.from_dict(rule_dict)
            # mise à jour des postérieurs + ema
            rule.observe_outcome(outcome)
            # persister (update si possible, sinon re-add)
            newd = rule.to_dict()
            if hasattr(self.arch.memory, "update_memory"):
                self.arch.memory.update_memory(newd)
            else:
                self.arch.memory.add_memory(newd)
            # trace
            try:
                self.arch.memory.add_memory({
                    "kind":"reward_event",
                    "rule_id": rule_id,
                    "outcome": outcome,
                    "ts": _now()
                })
            except Exception:
                pass

            try:
                reward_val = float((outcome or {}).get("reward", 0.5))
            except Exception:
                reward_val = 0.5
            self._reward_history.append(reward_val)
            window = int(getattr(self.arch, "social_calibration_window", 30))
            if window > 0 and len(self._reward_history) > window:
                self._reward_history = self._reward_history[-window:]
            alpha = float(getattr(self.arch, "social_calibration_alpha", 0.2))
            alpha = max(0.05, min(0.5, alpha))
            ema = None
            for r in self._reward_history:
                ema = r if ema is None else (alpha * r + (1.0 - alpha) * ema)
            if ema is not None:
                baseline = getattr(self, "_reward_perf_baseline", ema)
                self._reward_perf_baseline = (0.98 * baseline) + (0.02 * ema)
                drift = baseline - ema
                cooldown = max(10.0, float(getattr(self.arch, "social_calibration_cooldown", 30.0)))
                if drift > 0.04 and (_now() - getattr(self, "_last_calibration_ts", 0.0)) > cooldown:
                    weights = self.cfg.setdefault("weights", {})
                    adjust = min(0.02, drift * 0.1)
                    if adjust > 0.0:
                        weights["explicit_feedback"] = max(
                            0.25, float(weights.get("explicit_feedback", 0.45)) - adjust
                        )
                        weights["continue_dialogue"] = min(
                            0.22, float(weights.get("continue_dialogue", 0.10)) + adjust * 0.5
                        )
                        weights["valence"] = min(
                            0.22, float(weights.get("valence", 0.15)) + adjust * 0.5
                        )
                        self._last_calibration_ts = _now()
            return newd
        except Exception:
            return None
