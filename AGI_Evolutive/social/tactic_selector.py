# AGI_Evolutive/social/tactic_selector.py
# Sélection contextuelle d'une tactique sociale (bandit contextuel "diagonal LinUCB")
# Combine: match des prédicats, utilité attendue (effets postérieurs), reward EMA,
# anti-répétition (récence), garde-fous Policy, et incertitude (terme UCB).
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time, math, random

from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, ContextBuilder, Predicate, TacticSpec, clamp
)

def _now() -> float: return time.time()

# ---------------- Config par défaut ----------------
_DEFAULT_CFG = {
    "thresholds": {
        "match_min": 0.35,        # score de match (prédicats) minimal pour considérer une règle
        "score_min": 0.30,        # score final minimal pour appliquer une tactique
        "recent_sec": 300,        # pénalité si la règle a été utilisée < 300s
    },
    "weights": {
        "match":     0.40,        # poids du match symbolique
        "utility":   0.30,        # poids de l'utilité attendue (effets postérieurs)
        "ema_reward":0.20,        # poids du reward EMA appris (Social Critic)
        "bandit":    0.20,        # poids du terme UCB (incertitude / exploration)
        "recency":  -0.15,        # pénalité si réutilisation récente
    },
    "utility_weights": {          # pondération des effets dans expected_utility()
        "reduce_uncertainty": 0.35,
        "continue_dialogue":  0.25,
        "positive_valence":   0.25,
        "acceptance_marker":  0.15
    },
    "bandit": {
        "alpha": 0.6,             # intensité de l'exploration UCB
        "dim": 16                 # dimension du vecteur de contexte φ(s)
    },
    "epsilon": 0.08               # epsilon-greedy: chance de prendre la 2e meilleure
}

# ---------------- Petit bandit diagonal (sans numpy) ----------------
class DiagLinUCB:
    """
    LinUCB diagonalisé (pas d'inversion de matrice générale).
    - D : diag des A (variances par feature) -> liste de taille d
    - b : vecteur cumul des rewards pondérés par x -> liste de taille d
    Score UCB: x·θ + α * sqrt( sum_i (x_i^2 / D_i) )
    Update: D_i += x_i^2 ; b_i += r * x_i ; θ_i = b_i / D_i
    """
    def __init__(self, dim: int, alpha: float = 0.6):
        self.d = dim
        self.alpha = alpha
        self.D = [1.0] * dim   # variances init (identité)
        self.b = [0.0] * dim   # cumul
        self.n = 0             # nombre d'updates

    def score(self, x: List[float]) -> float:
        theta = [ (self.b[i] / self.D[i]) for i in range(self.d) ]
        # x·θ
        mean = sum( (x[i] * theta[i]) for i in range(self.d) )
        # incertitude
        conf = math.sqrt( sum( (x[i]*x[i]) / self.D[i] for i in range(self.d) ) )
        return float(mean + self.alpha * conf)

    def update(self, x: List[float], r01: float):
        self.n += 1
        for i in range(self.d):
            xi = x[i]
            self.D[i] += xi * xi
            self.b[i] += r01 * xi

    @staticmethod
    def load(dic: Dict[str, Any]) -> "DiagLinUCB":
        d = DiagLinUCB(dim=int(dic.get("dim", len(dic.get("D",[])) or _DEFAULT_CFG["bandit"]["dim"])),
                       alpha=float(dic.get("alpha", _DEFAULT_CFG["bandit"]["alpha"])))
        D = dic.get("D"); b = dic.get("b")
        if isinstance(D, list) and len(D) == d.d: d.D = [float(v) for v in D]
        if isinstance(b, list) and len(b) == d.d: d.b = [float(v) for v in b]
        d.n = int(dic.get("n", 0))
        return d

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.d, "alpha": self.alpha, "D": self.D, "b": self.b, "n": self.n}

# ---------------- Sélecteur ----------------
class TacticSelector:
    """
    Choisit une InteractionRule pertinente pour le contexte courant.
    Combine: match (prédicats), utilité attendue, reward EMA, bandit (incertitude), recency.
    Vérifie la Policy si dispo. Garde-fous: risque/contexte/persona.
    """
    def __init__(self, arch, cfg: Optional[Dict[str, Any]] = None):
        self.arch = arch
        self.cfg = cfg or getattr(arch, "tactic_selector_cfg", None) or _DEFAULT_CFG

    # ---------- Construction du contexte φ(s) ----------
    def _risk_num(self, level: str) -> float:
        if level in ("high","élevé"): return 1.0
        if level in ("medium","moyen"): return 0.5
        return 0.0

    def _one_hot(self, val: Optional[str], vocab: List[str]) -> List[float]:
        v = [0.0]*len(vocab)
        if val is None: return v
        try:
            i = vocab.index(val)
            v[i] = 1.0
        except ValueError:
            pass
        return v

    def _phi(self, ctx: Dict[str, Any], rule: Dict[str, Any]) -> List[float]:
        """
        Vecteur de contexte fixe (dim = cfg["bandit"]["dim"]).
        Compact, interprétable, sans dépendances.
        """
        dim = int(self.cfg["bandit"]["dim"])
        # Slots (≤ dim). Si moins, on padde par des zéros.
        # 0: intercept
        feats: List[float] = [1.0]

        # 1: match score (prédicats)
        # le match dépend de la règle -> recalcul ici
        try:
            r = InteractionRule.from_dict(rule)
            ms = r.match_score(ctx)
        except Exception:
            ms = 0.5
        feats.append(ms)

        # 2: risk level num
        feats.append(self._risk_num(str(ctx.get("risk_level","low")).lower()))

        # 3: persona alignment [0..1]
        feats.append(float(ctx.get("persona_alignment", 0.5)))

        # 4: valence ([-1..1] -> [0..1])
        val = float(ctx.get("valence", 0.0))
        feats.append((val + 1.0)/2.0)

        # 5: recence_usage (0..1; plus c'est haut, plus c'est récent/abus)
        feats.append(float(ctx.get("recence_usage", 0.0)))

        # 6..k: one-hot dialogue_act
        acts_vocab = ["question","compliment","insinuation","disagreement","confusion","ack","thanks","clarify","explain","statement"]
        feats += self._one_hot(ctx.get("dialogue_act"), acts_vocab)

        # 6+k..: one-hot implicature_hint (quelques hints)
        hints_vocab = ["sous-entendu","ironie","taquinerie","double-entendre"]
        feats += self._one_hot(ctx.get("implicature_hint"), hints_vocab)

        # tronquer/padder à dim
        if len(feats) < dim:
            feats += [0.0] * (dim - len(feats))
        elif len(feats) > dim:
            feats = feats[:dim]
        return feats

    # ---------- Politique / garde-fous ----------
    def _allowed_by_policy(self, rule: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[bool, str]:
        # 1) Risk gates simples si pas d'API policy
        risk = str(ctx.get("risk_level","low")).lower()
        tactic = (rule.get("tactic") or {}).get("name","")
        if risk == "high" and tactic in {"banter_leger"}:
            return False, "risk_high_no_banter"

        # 2) API policy si dispo
        pol = getattr(self.arch, "policy", None)
        if pol and hasattr(pol, "validate_tactic"):
            try:
                res = pol.validate_tactic(rule, ctx)
                if isinstance(res, dict):
                    dec = res.get("decision","allow")
                    return (dec == "allow"), f"policy:{dec}"
            except Exception:
                pass
        # 3) défaut
        return True, "ok"

    # ---------- Scoring combiné ----------
    def _score_rule(self, rule: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        W = self.cfg["weights"]
        th = self.cfg["thresholds"]
        now = _now()

        # a) match (prédicats)
        try:
            rr = InteractionRule.from_dict(rule)
            match = rr.match_score(ctx)
        except Exception:
            match = 0.0
        if match < float(th["match_min"]):
            return 0.0, {"skip":"low_match","match":round(match,3)}

        # b) utilité attendue (effets postérieurs)
        try:
            util = rr.expected_utility(weights=self.cfg.get("utility_weights"), exploration_bonus=0.0)
        except Exception:
            util = 0.5

        # c) reward EMA appris (Social Critic)
        ema = float(rule.get("ema_reward", 0.5))

        # d) bandit UCB (incertitude / exploration)
        bandit_state = rule.get("bandit") or {}
        blin = DiagLinUCB.load(bandit_state)
        x = self._phi(ctx, rule)
        ucb = blin.score(x)

        # Malus si même tactique utilisée très récemment
        last_used = float(rule.get("last_used_ts", 0.0))
        recency = max(0.0, 1.0 - ((now - last_used) / 60.0))
        diversity_penalty = 0.08 * recency
        ucb -= diversity_penalty

        # Coût optionnel si défini dans la règle
        cost = float(rule.get("cost", 0.0))
        ucb -= 0.10 * cost

        # e) pénalité de récence
        last_ts = float(rule.get("last_used_ts", 0.0))
        recent_pen = 0.0
        if last_ts > 0 and (now - last_ts) < float(th["recent_sec"]):
            recent_pen = 1.0  # sera multiplié par W["recency"] (négatif)

        # score final
        score = (
            float(W["match"])   * match +
            float(W["utility"]) * util  +
            float(W["ema_reward"])* ema +
            float(W["bandit"])  * ucb   +
            float(W["recency"]) * recent_pen
        )

        why = {
            "match": round(match,3),
            "utility": round(util,3),
            "ema_reward": round(ema,3),
            "ucb": round(ucb,3),
            "recent_pen": recent_pen,
            "tactic": (rule.get("tactic") or {}).get("name","")
        }
        return float(score), why

    # ---------- Pick principal ----------
    def pick(self, ctx: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Retourne (rule_dict, why) ou (None, None) si rien de suffisamment pertinent.
        Ne modifie pas la mémoire ici (le renderer tracera et marquera l'usage).
        """
        arch = self.arch
        ctx = ctx or ContextBuilder.build(arch)

        # 1) récupérer les règles depuis la mémoire
        try:
            rules = arch.memory.get_recent_memories(kind="interaction_rule", limit=400) or []
        except Exception:
            rules = []
        if not rules:
            return (None, None)

        # 2) filtrage simple Policy/garde-fous + scoring
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for r in rules:
            allow, reason = self._allowed_by_policy(r, ctx)
            if not allow:
                continue
            s, why = self._score_rule(r, ctx)
            if s <= 0.0:
                continue
            why["policy"] = reason
            scored.append((s, r, why))

        if not scored:
            return (None, None)

        # 3) tri & epsilon-greedy
        scored.sort(key=lambda t: t[0], reverse=True)
        best_s, best_r, best_why = scored[0]

        # epsilon: si la 2e est proche, on peut explorer
        eps = float(self.cfg.get("epsilon", 0.08))
        if len(scored) > 1 and random.random() < eps:
            s2, r2, w2 = scored[1]
            if s2 > float(self.cfg["thresholds"]["score_min"]):
                best_s, best_r, best_why = s2, r2, {**w2, "explore":"epsilon"}

        if best_s < float(self.cfg["thresholds"]["score_min"]):
            return (None, None)

        # 4) self-prompting court (explication interne) — pour traçabilité
        best_why["score"] = round(best_s,3)
        best_why["reason"] = f"match={best_why['match']}, util={best_why['utility']}, ema={best_why['ema_reward']}, ucb={best_why['ucb']}, policy={best_why.get('policy','ok')}"

        return (best_r, best_why)

    # ---------- Hook d'update bandit (appelé par Social Critic APRÈS feedback) ----------
    def bandit_update(self, rule_id: str, ctx: Dict[str, Any], reward01: float) -> Optional[Dict[str, Any]]:
        """
        Met à jour l'état du bandit de la règle (DiagLinUCB) à partir du contexte utilisé et du reward 0..1.
        Persiste en mémoire (update/add). Retourne la règle persistée (dict) ou None.
        """
        try:
            rules = self.arch.memory.get_recent_memories(kind="interaction_rule", limit=500) or []
        except Exception:
            rules = []
        rule = None
        for r in rules:
            if r.get("id") == rule_id:
                rule = r; break
        if not rule:
            return None

        x = self._phi(ctx, rule)
        blin = DiagLinUCB.load(rule.get("bandit") or {})
        blin.update(x, float(reward01))
        rule["bandit"] = blin.to_dict()

        # légère adaptation de cooldown en fonction du reward (punitif si très bas)
        cd = float(rule.get("cooldown", 0.0))
        if reward01 < 0.25:
            cd = min(1.0, cd + 0.15)
        elif reward01 > 0.75:
            cd = max(0.0, cd * 0.9)
        rule["cooldown"] = cd

        if hasattr(self.arch.memory, "update_memory"):
            self.arch.memory.update_memory(rule)
        else:
            self.arch.memory.add_memory(rule)
        return rule


# ---------------------------------------------------------------------------
# Lightweight style bandit helper (for macro selection, optional usage)


class _ThompsonBandit:
    def __init__(self) -> None:
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)

    def sample(self, arm: str) -> float:
        a = self.alpha[arm]
        b = self.beta[arm]
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        return x / (x + y) if (x + y) else 0.5

    def update(self, arm: str, reward: float) -> None:
        if reward > 0:
            self.alpha[arm] += reward
        elif reward < 0:
            self.beta[arm] += -reward


class StyleMacroBandit:
    """Petit bandit Thompson-sampling pour choisir un macro style."""

    DEFAULT_ARMS = ["sobre", "taquin", "coach", "deadpan"]

    def __init__(self, arms: Optional[List[str]] = None) -> None:
        self.arms = arms or list(self.DEFAULT_ARMS)
        self._bandit = _ThompsonBandit()
        self._last_arm: Optional[str] = None

    def pick(self, context: Optional[Dict[str, Any]] = None) -> str:
        scored = [(self._bandit.sample(arm), arm) for arm in self.arms]
        scored.sort(key=lambda item: item[0], reverse=True)
        self._last_arm = scored[0][1]
        return self._last_arm

    def feedback(self, reward: float, arm: Optional[str] = None) -> None:
        choice = arm or self._last_arm
        if not choice:
            return
        self._bandit.update(choice, float(reward))
