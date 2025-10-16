from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Dict, Any, List, Optional, Tuple

import random, copy, json, os, time

from AGI_Evolutive.utils.jsonsafe import json_sanitize


def rag_quality_signal(signals: dict) -> float:
    if not signals:
        return 0.0
    top1 = float(signals.get("rag_top1", 0.0))
    mean = float(signals.get("rag_mean", 0.0))
    div  = float(signals.get("rag_diversity", 0.0))
    n    = float(signals.get("rag_docs", 0.0))
    return max(0.0, min(1.0, 0.45*top1 + 0.35*mean + 0.15*div + 0.05*min(n/5.0,1.0)))


class PolicyEngine:
    """Stores lightweight policy directives and strategy hints."""

    def __init__(self, path: str = "data/policy.json"):
        self.path = path
        self.state: Dict[str, Any] = {
            "drive_targets": {},
            "hints": []
        }
        self._load()
        # --- Policy telemetry / stats (fichier séparé, on ne touche pas à path)
        self._stats_path = "runtime/policy_stats.json"
        self._stats = defaultdict(lambda: {"s": 0, "n": 0})  # succès/essais par type de proposition
        self._last_confidence = 0.55
        self.last_decision: Optional[Dict[str, Any]] = None

        # charge si existe (best-effort)
        try:
            if os.path.exists(self._stats_path):
                with open(self._stats_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for k, v in raw.items():
                        self._stats[k] = {"s": int(v.get("s", 0)), "n": int(v.get("n", 0))}
        except Exception:
            pass

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.state = json.load(fh)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(json_sanitize(self.state), fh, ensure_ascii=False, indent=2)

    def _stats_save(self) -> None:
        try:
            directory = os.path.dirname(self._stats_path) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump({k: dict(v) for k, v in self._stats.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _key(self, proposal: Dict[str, Any]) -> str:
        return f"{proposal.get('type')}|{'/'.join(map(str, proposal.get('path', [])))}"

    def register_outcome(self, proposal: Dict[str, Any], success: bool) -> None:
        """À appeler APRES exécution d’une proposition (true/false)."""
        try:
            k = self._key(proposal)
            self._stats[k]["n"] += 1
            self._stats[k]["s"] += 1 if success else 0
            self._stats_save()
        except Exception:
            pass

    def _wilson_lower_bound(self, s: int, n: int, z: float = 1.96) -> float:
        """Borne inférieure de Wilson (intervalle de confiance) = prudente."""
        if n <= 0:
            return 0.55  # prior doux
        p = s / n
        denom = 1 + (z * z) / n
        centre = p + (z * z) / (2 * n)
        margin = z * sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
        return max(0.0, min(1.0, (centre - margin) / denom))

    def _freq_conf(self, proposal: Dict[str, Any]) -> float:
        st = self._stats[self._key(proposal)]
        return self._wilson_lower_bound(st["s"], st["n"])

    def _stability_probe(self, proposer, homeo, runs: int = 5, jitter: float = 0.03) -> float:
        """Accord du top-1 sous petites perturbations des drives : 1 = très stable."""
        try:
            if proposer is None or homeo is None or not hasattr(proposer, "run_once_now"):
                return 0.5
            state = getattr(homeo, "state", {})
            if not isinstance(state, dict):
                return 0.5
            drives = state.get("drives", {})
            if not isinstance(drives, dict):
                return 0.5
            base = dict(drives)
            tops = []
            for _ in range(runs):
                perturbed = {k: max(0.0, min(1.0, v + random.uniform(-jitter, jitter))) for k, v in base.items()}
                drives.update(perturbed)
                props = proposer.run_once_now() or []
                tops.append(str(props[:1]))
            drives.update(base)
            most = max(tops.count(t) for t in set(tops))
            return most / runs
        except Exception:
            return 0.5

    def _value_and_risk(self, planner, proposal: Dict[str, Any], trials: int = 5) -> Tuple[float, float]:
        """Utilise Planner.simulate_action si dispo : renvoie (valeur≈[0..1], risque≈[0..1])."""
        try:
            if planner is None or not hasattr(planner, "simulate_action"):
                return 0.6, 0.25
            vals: List[float] = []
            for _ in range(trials):
                try:
                    v = float(planner.simulate_action(proposal))
                    vals.append(v)
                except Exception:
                    pass
            if not vals:
                return 0.6, 0.25
            m = sum(vals) / len(vals)
            # normalisation sigmoïde grossière pour valeur
            val = 1.0 / (1.0 + (2.718281828) ** (-m))
            # écart-type (borné)
            mu = m
            std = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
            risk = min(1.0, std / 5.0)
            return float(val), float(risk)
        except Exception:
            return 0.6, 0.25

    def confidence_for(self,
                       proposal: Dict[str, Any],
                       *,
                       proposer=None,
                       homeo=None,
                       planner=None,
                       ctx: Optional[Dict[str, Any]] = None) -> float:
        """Score final 0..1 : mélange fréquence (Wilson), stabilité, valeur, croyance, avec pénalité risque/novelty."""
        ctx = ctx or {}
        freq = self._freq_conf(proposal)                    # 0..1
        stab = self._stability_probe(proposer, homeo)       # 0..1
        val, risk = self._value_and_risk(planner, proposal) # 0..1, 0..1
        belief = float(ctx.get("belief_confidence", 0.6))   # 0..1 (exposé par SelfModel si tu l’as)
        novelty_fam = float(ctx.get("novelty_familiarity", 0.7))  # 0..1 (1 = familier, 0 = inédit)

        w = {"freq": 0.35, "stab": 0.25, "val": 0.25, "belief": 0.15}
        raw = w["freq"] * freq + w["stab"] * stab + w["val"] * val + w["belief"] * belief
        penalty = 0.20 * risk + 0.10 * (1.0 - novelty_fam)
        conf = max(0.0, min(1.0, raw - penalty))
        self._last_confidence = conf
        return conf

    def confidence(self) -> float:
        """Confiance globale (agrégée) — fallback si le test veut une API simple."""
        if not self._stats:
            return self._last_confidence
        vals = [self._wilson_lower_bound(v["s"], v["n"]) for v in self._stats.values()]
        vals.sort()
        median = vals[len(vals) // 2]
        return float(0.5 * median + 0.5 * self._last_confidence)

    def explain(self, proposal: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
        """Renvoie les composantes (freq/stab/val/risk/belief/novelty) + score final."""
        if proposal is None:
            return {
                "confidence": self.confidence(),
                "note": "global"
            }
        proposer = kw.get("proposer")
        homeo = kw.get("homeo")
        planner = kw.get("planner")
        ctx = kw.get("ctx", {})
        freq = self._freq_conf(proposal)
        stab = self._stability_probe(proposer, homeo)
        val, risk = self._value_and_risk(planner, proposal)
        belief = float(ctx.get("belief_confidence", 0.6))
        novelty_fam = float(ctx.get("novelty_familiarity", 0.7))
        w = {"freq": 0.35, "stab": 0.25, "val": 0.25, "belief": 0.15}
        raw = w["freq"] * freq + w["stab"] * stab + w["val"] * val + w["belief"] * belief
        penalty = 0.20 * risk + 0.10 * (1.0 - novelty_fam)
        conf = max(0.0, min(1.0, raw - penalty))
        return {
            "components": {"freq": freq, "stab": stab, "val": val, "risk": risk, "belief": belief, "novelty_familiarity": novelty_fam},
            "weights": w,
            "penalty": penalty,
            "confidence": conf,
        }
    def adjust_drive_target(self, drive: str, target: float):
        self.state.setdefault("drive_targets", {})[drive] = float(max(0.0, min(1.0, target)))
        self.state.setdefault("history", []).append({
            "ts": time.time(),
            "event": "drive_target",
            "drive": drive,
            "target": target
        })
        self.state["history"] = self.state["history"][-200:]
        self._save()

    def register_hint(self, hint: Dict[str, Any]):
        entry = dict(hint)
        entry["ts"] = time.time()
        self.state.setdefault("hints", []).append(entry)
        self.state["hints"] = self.state["hints"][-100:]
        self._save()

    def validate_tactic(self, rule: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stub pour validation de tactique : autorise par défaut."""
        return {"decision": "allow", "reason": "default_allow"}

    def recent_frictions(self, window_sec: int = 600) -> int:
        """Stub pour comptage des frictions récentes."""
        return 0

    def validate_proposal(self, proposal: Dict[str, Any], self_state: Dict[str, Any]) -> Dict[str, Any]:
        path = proposal.get("path", [])
        if not path:
            return {"decision": "deny", "reason": "path manquant"}

        if path[0] == "core_immutable":
            return {"decision": "deny", "reason": "noyau protégé"}

        if path == ["identity", "name"] and isinstance(proposal.get("value"), str) and len(proposal["value"]) > 20:
            return {"decision": "needs_human", "reason": "changement identité important"}

        return {"decision": "allow", "reason": "OK"}

    def compute_frame_utility(
        self,
        frame: Any,
        *,
        weights: Optional[Dict[str, float]] = None,
        components: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute aggregate utility for a conversational frame."""

        def _clamp01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        def _collect_from(obj: Any, keys: List[str]) -> Dict[str, float]:
            collected: Dict[str, float] = {}
            for key in keys:
                source = None
                if isinstance(obj, dict):
                    source = obj.get(key)
                else:
                    source = getattr(obj, key, None)
                if isinstance(source, dict):
                    for name, val in source.items():
                        try:
                            collected[name] = float(val)
                        except (TypeError, ValueError):
                            continue
            return collected

        comp_sources = _collect_from(frame, ["utilities", "utility_components", "scores", "drives"])
        if components:
            for key, val in components.items():
                try:
                    comp_sources[key] = float(val)
                except (TypeError, ValueError):
                    continue

        w_sources = _collect_from(frame, ["weights", "drive_weights", "priorities"])
        if weights:
            for key, val in weights.items():
                try:
                    w_sources[key] = float(val)
                except (TypeError, ValueError):
                    continue

        U_survive = _clamp01(
            comp_sources.get("survive", comp_sources.get("Survive", 0.5))
        )
        U_evolve = _clamp01(
            comp_sources.get("evolve", comp_sources.get("Evolve", 0.5))
        )
        U_interact = _clamp01(
            comp_sources.get("interact", comp_sources.get("Interact", 0.5))
        )

        w_survive = w_sources.get("survive", w_sources.get("Survive", 1.0))
        w_evolve = w_sources.get("evolve", w_sources.get("Evolve", 1.0))
        w_interact = w_sources.get("interact", w_sources.get("Interact", 1.0))

        try:
            w_survive = float(w_survive)
        except (TypeError, ValueError):
            w_survive = 1.0
        try:
            w_evolve = float(w_evolve)
        except (TypeError, ValueError):
            w_evolve = 1.0
        try:
            w_interact = float(w_interact)
        except (TypeError, ValueError):
            w_interact = 1.0

        total_w = w_survive + w_evolve + w_interact
        if total_w <= 0:
            w_survive = w_evolve = w_interact = 1.0 / 3.0
        else:
            w_survive /= total_w
            w_evolve /= total_w
            w_interact /= total_w

        rq = rag_quality_signal(getattr(frame, "signals", {}))
        if isinstance(frame, dict):
            rq = rag_quality_signal(frame.get("signals", {}))

        # “Survivre” → éviter erreurs si support faible
        U_survive *= (0.7 + 0.6*rq)

        # “Évoluer” → exploiter lorsque support solide
        U_evolve  *= (0.8 + 0.5*rq)

        # “Interagir” → si support moyen/faible, favoriser clarification
        if rq < 0.35:
            U_interact *= 0.85
        elif rq < 0.6:
            U_interact *= 1.0
        else:
            U_interact *= 1.1

        U = w_survive*U_survive + w_evolve*U_evolve + w_interact*U_interact
        return max(0.0, min(1.0, U))

    def decide(self,
               proposals: List[Dict[str, Any]],
               self_state: Optional[Dict[str, Any]] = None,
               *,
               proposer=None,
               homeo=None,
               planner=None,
               ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sélectionne la meilleure proposition + confiance, avec abstention contrôlée."""
        self_state = self_state or {}
        ctx = ctx or {}
        if not proposals:
            self.last_decision = {"decision": "noop", "reason": "no proposals", "confidence": 0.5}
            return self.last_decision

        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        for p in proposals:
            gate = self.validate_proposal(p, self_state)  # conserve ta logique actuelle (allow/deny/needs_human)
            if gate.get("decision") == "deny":
                continue

            conf = self.confidence_for(p, proposer=proposer, homeo=homeo, planner=planner, ctx=ctx)
            # bonus léger sur actions "act" / malus sur "needs_human"
            bonus = 0.05 if p.get("type") in {"act", "do_step"} else 0.0
            if gate.get("decision") == "needs_human":
                conf -= 0.15
            final = max(0.0, min(1.0, conf + bonus))

            scored.append((final, p, gate, {"conf": conf, "bonus": bonus}))

        if not scored:
            self.last_decision = {"decision": "noop", "reason": "all denied", "confidence": 0.45}
            return self.last_decision

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_p, best_gate, meta = scored[0]

        if best_gate.get("decision") == "needs_human":
            self.last_decision = {
                "decision": "needs_human",
                "reason": best_gate.get("reason", "human validation required"),
                "confidence": best_score,
                "proposal": best_p,
                "policy_reason": best_gate.get("reason", ""),
                "meta": meta,
            }
            return self.last_decision

        # seuil d’abstention (risk-coverage simple)
        ABSTAIN = 0.58
        if best_score < ABSTAIN:
            self.last_decision = {
                "decision": "needs_human",
                "reason": "low confidence",
                "confidence": best_score,
                "proposal": best_p,
                "policy_reason": best_gate.get("reason", ""),
                "meta": meta,
            }
            return self.last_decision

        self.last_decision = {
            "decision": "apply",
            "proposal": best_p,
            "policy_reason": best_gate.get("reason", ""),
            "confidence": best_score,
            "meta": meta,
        }
        return self.last_decision
