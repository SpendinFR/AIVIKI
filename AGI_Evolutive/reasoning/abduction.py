from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple, Callable
import datetime as dt
import re

from .structures import CausalStore, DomainSimulator, HTNPlanner, SimulationResult, TaskNode

@dataclass
class Hypothesis:
    label: str
    explanation: str
    score: float
    priors: Dict[str, float]
    evidence: List[str]
    ask_next: Optional[str] = None
    causal_support: List[str] = field(default_factory=list)
    simulations: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[List[str]] = None

class AbductiveReasoner:
    """
    Pipeline:
      - generators -> candidats
      - scoring -> score(H) par priors + matches + beliefs
      - causalité explicite (CausalStore)
      - vérification par simulation (DomainSimulator)
      - planification HTN (HTNPlanner)
      - active question (politique d'entropie)
      - calibration (log prédiction) gérée côté arch
    """

    def __init__(
        self,
        beliefs,
        user_model,
        generators: Optional[List[Callable[[str], List[Tuple[str, str]]]]] = None,
        causal_store: Optional[CausalStore] = None,
        simulator: Optional[DomainSimulator] = None,
        planner: Optional[HTNPlanner] = None,
        question_policy: Optional["EntropyQuestionPolicy"] = None,
    ):
        self.beliefs = beliefs
        self.user = user_model
        self.generators = generators or [self._g_default]
        self.causal_store = causal_store or CausalStore()
        self.simulator = simulator or DomainSimulator()
        self.planner = planner or HTNPlanner()
        self.question_policy = question_policy or EntropyQuestionPolicy()

        if not self.planner.has_template("diagnostic_general"):
            self._register_default_plan()

    def _g_default(self, text: str) -> List[Tuple[str, str]]:
        t = text.lower()
        if "jaune" in t and ("ongle" in t or "doigt" in t):
            return [
                ("cheddar", "résidu de fromage fondu"),
                ("curcuma", "résidu d'épice"),
                ("nicotine", "tache nicotine"),
                ("infection", "sécrétion ou mycose"),
                ("autre", "autre cause"),
            ]
        # fallback neutre
        return [("inconnu", "cause inconnue faute d'indices")]

    def _time(self) -> dt.datetime:
        return dt.datetime.now()

    def _score(self, label: str, text: str) -> Tuple[float, Dict[str,float], List[str]]:
        t = text.lower(); now = self._time()
        pri = 0.5
        ev: List[str] = []
        # priors utilisateur
        pri = max(pri, self.user.prior(label)) if hasattr(self.user, "prior") else pri
        pri = min(1.0, pri + self.user.routine_bias(label, now)) if hasattr(self.user, "routine_bias") else pri
        # beliefs
        boost = 0.0
        for b in self.beliefs.query(subject="user", relation="likes", min_conf=0.6):
            if b.value in {label, f"{label}_related"}:
                boost += 0.15
        # matches (features toy; à étendre par domaine)
        matches = 0.0
        if "midi" in t: matches += 0.05
        if "réchauff" in t or "micro" in t: matches += 0.05
        if label in t: matches += 0.2
        score = max(0.0, min(1.0, 0.5*pri + boost + matches))
        return score, {"pri": pri, "boost": boost, "matches": matches}, ev

    def generate(self, observation_text: str) -> List[Hypothesis]:
        cands: List[Tuple[str, str]] = []
        for g in self.generators:
            try:
                cands.extend(g(observation_text))
            except Exception:
                continue
        # dedup
        uniq: Dict[str, str] = {}
        for lab, why in cands:
            uniq.setdefault(lab, why)
        hyps: List[Hypothesis] = []
        for lab, why in uniq.items():
            s, pri, ev = self._score(lab, observation_text)
            causal_support = self._causal_support(lab, observation_text)
            simulations = self._run_simulations(lab, observation_text)
            plan = self._plan_validation(lab, observation_text)
            hyps.append(
                Hypothesis(
                    label=lab,
                    explanation=why,
                    score=s,
                    priors=pri,
                    evidence=ev,
                    causal_support=causal_support,
                    simulations=simulations,
                    plan=plan,
                )
            )
        hyps.sort(key=lambda h: h.score, reverse=True)
        # active question si nécessaire
        if self.question_policy:
            question = self.question_policy.suggest(hyps, observation_text)
            if question and hyps:
                hyps[0].ask_next = question
        return hyps[:5]

    def _craft_question(self, a: str, b: str) -> str:
        # squelette générique
        return f"As-tu un indice concret qui pointerait plutôt vers {a} (trace, odeur, contexte horaire) ou {b} ?"

    # -- causal reasoning -------------------------------------------------
    def _register_default_plan(self) -> None:
        diagnostic = TaskNode(
            name="diagnostic_general",
            preconditions=["observation"],
            actions=[
                "Collecter un indice supplémentaire fiable.",
                "Comparer les hypothèses en fonction des conditions observées.",
                "Chercher une contre-preuve rapide.",
            ],
            postconditions=["hypothesis_validated"],
        )
        self.planner.register_template("diagnostic_general", diagnostic)

    def _causal_support(self, label: str, text: str) -> List[str]:
        if not self.causal_store:
            return []
        lower = text.lower()
        tokens = re.findall(r"[a-zàâäéèêëîïôöùûüç]+", lower)
        observed_effects = {token for token in tokens if len(token) > 3}
        supports: List[str] = []
        for link in self.causal_store.get_effects(label):
            observed = link.effect.lower() in observed_effects or link.effect.lower() in lower
            status = "observé" if observed else "à vérifier"
            msg = f"{label} ⇒ {link.effect} (force {link.strength:.2f}, {status})"
            if link.conditions:
                msg += f" | conditions: {', '.join(link.conditions)}"
            supports.append(msg)
            test = self.causal_store.test_relation(
                cause=label,
                effect=link.effect,
                context={"observation": text},
            )
            if test["supported"]:
                if test["unsatisfied_conditions"]:
                    cond_status = f"conditions manquantes: {', '.join(test['unsatisfied_conditions'])}"
                else:
                    cond_status = "conditions ok"
                supports.append(f"Test causal: {label} ⇒ {link.effect} ({cond_status})")
        for link in self.causal_store.get_causes(label):
            if link.cause.lower() in lower:
                supports.append(
                    f"Observation compatible: {link.cause} pourrait provoquer {label} (force {link.strength:.2f})"
                )
        return supports

    def _run_simulations(self, label: str, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        scenario = {"observation": text, "hypothesis": label}
        sim_result = self.simulator.simulate(label, scenario)
        results.append(
            {
                "domain": label,
                "timestamp": self._time().isoformat(),
                **sim_result.to_dict(),
            }
        )
        if not sim_result.success:
            coherence = label.lower() in text.lower()
            mental = SimulationResult(
                success=coherence,
                outcome=(
                    "Le scénario est cohérent avec les éléments textuels."
                    if coherence
                    else "Le scénario semble faible : aucun indice direct dans l'observation."
                ),
                details={"method": "mental_check"},
            )
            results.append(
                {
                    "domain": "mental",
                    "timestamp": self._time().isoformat(),
                    **mental.to_dict(),
                }
            )
        return results

    def _plan_validation(self, label: str, text: str) -> Optional[List[str]]:
        context = {"observation": text, "hypothesis": label}
        goal = f"valider_{label}"
        plan = self.planner.plan(goal, context=context)
        if not plan:
            plan = self.planner.plan("diagnostic_general", context=context)
        if not plan:
            plan = [
                f"Observer de près les indices associés à {label}.",
                "Comparer avec au moins une hypothèse alternative.",
                "Collecter une nouvelle donnée ciblée.",
            ]
        return plan


class EntropyQuestionPolicy:
    """Active questioning policy based on entropy reduction heuristics."""

    def suggest(self, hypotheses: List[Hypothesis], observation: str) -> Optional[str]:
        if len(hypotheses) < 2:
            return None
        top = hypotheses[:4]
        scores = [max(h.score, 1e-4) for h in top]
        total = float(sum(scores))
        if total == 0.0:
            return None
        probs = [score / total for score in scores]
        best_pair: Optional[Tuple[Hypothesis, Hypothesis]] = None
        best_gain = 0.0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                gain = 2.0 * min(probs[i], probs[j])
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (top[i], top[j])
        if not best_pair or best_gain < 0.05:
            return None
        a, b = best_pair
        discriminants: List[str] = []
        if a.causal_support:
            discriminants.append(a.causal_support[0])
        if b.causal_support:
            discriminants.append(b.causal_support[0])
        base = f"Quelle observation permettrait de trancher entre {a.label} et {b.label}?"
        if discriminants:
            base = (
                f"Quel indice vérifierait {a.label} ({discriminants[0]}) ou {b.label} ({discriminants[-1]}) ?"
            )
        info = f"(gain info ≈ {best_gain:.2f})"
        return f"{base} {info}"
