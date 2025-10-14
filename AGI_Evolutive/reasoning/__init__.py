"""
Reasoning Engine v1
- Orchestration multi-stratégies (décomposition → récupération → hypothèses → auto-vérification)
- Historique structuré (recent_inferences, learning_trajectory)
- Statistiques pour la métacognition (average_confidence, strategy_preferences)
- Tolerant aux modules absents (perception/mémoire)
"""

import time
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Callable

from .strategies import (
    ReasoningStrategy,
    DecompositionStrategy,
    EvidenceRetrievalStrategy,
    HypothesisRankingStrategy,
    SelfCheckStrategy,
)


class ReasoningSystem:
    def __init__(self, arch, memory, perception):
        self.arch = arch
        self.memory = memory
        self.perception = perception

        # Historique exploitable par métacognition et autres systèmes
        self.reasoning_history: Dict[str, Any] = {
            "recent_inferences": deque(maxlen=100),
            "learning_trajectory": [],   # [{ts, confidence}]
            "last_answer": "",
        }

        # Catalogue de stratégies
        self.strategies: List[ReasoningStrategy] = [
            DecompositionStrategy(),
            EvidenceRetrievalStrategy(),
            HypothesisRankingStrategy(),
            SelfCheckStrategy(),
        ]

        # Comptage d’usage par stratégie
        self.strategy_usage = defaultdict(int)

    # ----------- API externe principale -----------
    def reason(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Lance un mini-raisonnement explicable.
        Retourne:
        {
          "answer": str,
          "confidence": float,
          "trace": [{"strategy": name, "notes": str, "questions": [...], "time": float}],
          "support": [str],
          "meta": {"complexity": float, "duration": float}
        }
        """
        t0 = time.time()
        trace: List[Dict[str, Any]] = []
        support: List[str] = []
        proposals: List[Dict[str, Any]] = []
        context = dict(context or {})

        # Outils (passerelles) mis à disposition des stratégies
        toolkit: Dict[str, Callable] = {
            "retrieve_fn": self._retrieve_hits_safe
        }

        # 1) Décomposition
        step = self._run_strategy(self.strategies[0], prompt, context, toolkit)
        trace.append(step["trace"])
        if step["support"]:
            support.extend(step["support"])
        context.update(step["context"])

        # 2) Récupération
        step = self._run_strategy(self.strategies[1], prompt, context, toolkit)
        trace.append(step["trace"])
        if step["support"]:
            support.extend(step["support"])
        context.update(step["context"])

        # 3) Hypothèses
        step = self._run_strategy(self.strategies[2], prompt, context, toolkit)
        trace.append(step["trace"])
        proposals = step["context"].get("proposals", [])
        context.update(step["context"])

        # 4) Auto-vérification (utilise last_answer)
        context["last_answer"] = self.reasoning_history.get("last_answer", "")
        context["proposals"] = proposals
        step = self._run_strategy(self.strategies[3], prompt, context, toolkit)
        trace.append(step["trace"])
        proposals = step["context"].get("proposals", proposals)

        # Sélection finale
        best = max(proposals, key=lambda p: p.get("confidence", 0.0)) if proposals else {
            "answer": "Je manque d’éléments pour conclure. Pouvez-vous préciser le but et les contraintes ?",
            "confidence": 0.3,
            "support": []
        }

        duration = time.time() - t0
        complexity = self._estimate_complexity(trace)
        answer = best.get("answer", "")
        confidence = float(best.get("confidence", 0.0))
        support = best.get("support", []) or support[:3]

        # Enregistrement historique (utilisé par métacognition)
        self._record_inference(
            question=prompt,
            solution=answer,
            final_confidence=confidence,
            reasoning_time=duration,
            complexity=complexity,
            strategies=[t["strategy"] for t in trace],
        )

        result = {
            "answer": answer,
            "confidence": confidence,
            "trace": trace,          # résumé des étapes, pas de chaîne de pensée sensible
            "support": support,
            "meta": {
                "complexity": complexity,
                "duration": duration,
            }
        }
        # Mémoriser la dernière réponse pour l'auto-vérification future
        self.reasoning_history["last_answer"] = answer
        return result

    # ----------- Intégration métacognitive -----------
    def get_reasoning_stats(self) -> Dict[str, Any]:
        recent = list(self.reasoning_history["recent_inferences"])
        avg_conf = sum(x.get("final_confidence", 0.0) for x in recent) / max(1, len(recent))
        total = sum(self.strategy_usage.values()) or 1
        prefs = {name: count / total for name, count in self.strategy_usage.items()}
        return {
            "average_confidence": float(avg_conf),
            "strategy_preferences": prefs,
            "inference_count": len(recent),
        }

    # ----------- Stratégie runner + outils -----------
    def _run_strategy(self, strategy: ReasoningStrategy, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]):
        try:
            out = strategy.apply(prompt, context, toolkit) or {}
        except Exception as e:
            out = {"notes": f"Erreur stratégie {strategy.name}: {e}", "proposals": [], "questions": [], "cost": 0.0, "time": 0.0}

        self.strategy_usage[strategy.name] += 1

        # Normalisation
        notes = out.get("notes", "")
        proposals = out.get("proposals", [])
        questions = out.get("questions", [])
        support = out.get("support", [])

        # Le contexte agrège questions/propositions/support
        new_context = dict(context)
        if proposals:
            new_context["proposals"] = (new_context.get("proposals", []) + proposals)
        if questions:
            new_context["subquestions"] = (new_context.get("subquestions", []) + questions)
        if support:
            # stock commun de support pour stratégies suivantes
            new_context["support"] = (new_context.get("support", []) + support)

        trace_item = {
            "strategy": strategy.name,
            "notes": notes,
            "questions": questions,
            "time": float(out.get("time", 0.0)),
        }
        return {"trace": trace_item, "context": new_context, "support": support}

    def _retrieve_hits_safe(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        mem = getattr(self.arch, "memory", None)
        retr = getattr(mem, "retrieval", None) if mem else None
        try:
            if retr:
                return retr.index.search_text(query, top_k=top_k)  # renvoie docs internes (id, score, text, meta)
        except Exception:
            pass
        return []

    # ----------- Historisation / métriques -----------
    def _estimate_complexity(self, trace: List[Dict[str, Any]]) -> float:
        """
        Complexité heuristique: nb d’étapes + présence de sous-questions + support.
        Normalisé [0,1].
        """
        steps = len(trace)
        sq = sum(len(t.get("questions", []) or []) for t in trace)
        comp = 0.2 * min(steps, 6) + 0.1 * min(sq, 10)
        return float(max(0.0, min(1.0, comp)))

    def _record_inference(self, question: str, solution: str, final_confidence: float,
                          reasoning_time: float, complexity: float, strategies: List[str]):
        rec = {
            "timestamp": time.time(),
            "question": question,
            "solution": solution,
            "final_confidence": float(final_confidence),
            "reasoning_time": float(reasoning_time),
            "complexity": float(complexity),
            "strategies": list(strategies),
        }
        self.reasoning_history["recent_inferences"].append(rec)
        trajectory: List[Dict[str, Any]] = self.reasoning_history["learning_trajectory"]
        trajectory.append({
            "ts": rec["timestamp"],
            "confidence": float(final_confidence),
        })
        if len(trajectory) > 200:
            del trajectory[0]
# reasoning/__init__.py
from collections import deque
from typing import Any, Dict, List, Optional
import time

from reasoning.structures import Hypothesis, Test, Evidence, episode_record


class ReasoningSystem:
    """
    Raisonner ≠ générer du texte : ici on
      - fabrique des hypothèses (intention / stratégie)
      - estime leur valeur (heuristiques + signaux de style)
      - propose des tests concrets
      - journalise un "épisode" traçable
      - maintient des stats + trajectoire d'apprentissage
    """

    def __init__(self, architecture, memory_system=None, perception_system=None):
        self.arch = architecture
        self.memory = memory_system
        self.perception = perception_system

        # Historique structuré (attendu par la méta)
        self.reasoning_history: Dict[str, Any] = {
            "recent_inferences": deque(maxlen=200),
            "learning_trajectory": deque(maxlen=500),
            "errors": deque(maxlen=200),
            "stats": {
                "n_episodes": 0,
                "avg_confidence": 0.50,
                "strategy_preferences": {
                    "abduction": 0.33,
                    "deduction": 0.34,
                    "analogy": 0.33
                }
            }
        }

    # ------------------- API publique -------------------

    def reason_about(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Raisonner sur 'prompt' → retourne un dict:
          - summary (texte court non-générique)
          - chosen_hypothesis (str)
          - tests (list[str])
          - final_confidence (float)
          - appris (list[str])
          - prochain_test (str)
          - episode (dict sérialisé)
        """
        t0 = time.time()
        ctx = context or {}

        # 1) Sélection de stratégie de haut niveau (très simple mais efficace)
        strategy = self._pick_strategy(prompt)

        # 2) Génère 3 hypothèses d’intention
        hypos = self._make_hypotheses(prompt, strategy=strategy)

        # 3) Scoring des hypothèses (heuristiques + style policy si dispo)
        scores = self._score_hypotheses(prompt, hypos, strategy)
        chosen_idx = max(range(len(scores)), key=lambda i: scores[i])

        # 4) Proposer 2–3 tests concrets
        tests = self._propose_tests(prompt, strategy=strategy)

        # 5) "Exécution" symbolique (ici: micro-inférence + evidence text)
        note = self._simulate_micro_inference(prompt, strategy, hypos[chosen_idx])
        evidence = Evidence(notes=note, confidence=min(0.55 + 0.15 * scores[chosen_idx], 0.95))

        # 6) Post-traitement : confiance, complexité, durée
        reasoning_time = max(0.05, time.time() - t0)
        complexity = self._estimate_complexity(prompt, strategy, hypos)

        final_conf = float(min(0.9, 0.45 + 0.4 * scores[chosen_idx] + 0.10 * (1.0 - complexity)))

        # 7) Journalisation épisode (JSONL via arch.logger)
        ep = episode_record(
            user_msg=prompt,
            hypotheses=hypos,
            chosen=chosen_idx,
            tests=tests,
            evidence=evidence,
            result_text=evidence.notes,
            final_confidence=final_conf
        )
        ep["reasoning_time"] = reasoning_time
        ep["complexity"] = complexity
        ep["strategy"] = strategy

        self._push_episode(ep)

        # 8) Résumé exploitable pour la réponse
        summary = self._make_readable_summary(strategy, hypos, chosen_idx, tests, evidence, final_conf)

        # 9) Apprentissage local simple (trajectoire)
        self._learn(final_conf, strategy)

        # 10) Log externe
        try:
            if hasattr(self.arch, "logger") and self.arch.logger:
                self.arch.logger.write("reasoning.episode", episode=ep)
        except Exception:
            pass

        # 11) Sortie pour l’architecture
        return {
            "summary": summary,
            "chosen_hypothesis": hypos[chosen_idx].content,
            "tests": [t.description for t in tests],
            "final_confidence": final_conf,
            "appris": [
                f"Stratégie={strategy}, complexité≈{complexity:.2f}",
                "Toujours relier hypothèse→test→évidence (traçabilité)."
            ],
            "prochain_test": tests[0].description if tests else "—",
            "episode": ep
        }

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Utilisé par la méta/monitoring."""
        st = self.reasoning_history["stats"]
        recents = list(self.reasoning_history["recent_inferences"])
        if recents:
            st["avg_confidence"] = sum(x.get("final_confidence", 0.5) for x in recents) / len(recents)
        return {
            "episodes": st["n_episodes"],
            "average_confidence": float(st.get("avg_confidence", 0.5)),
            "strategy_preferences": dict(st.get("strategy_preferences", {}))
        }

    # ------------------- internes -------------------

    def _pick_strategy(self, prompt: str) -> str:
        p = (prompt or "").lower()
        if "pourquoi" in p or "why" in p:
            return "abduction"
        if "comment" in p or "how" in p:
            return "deduction"
        if "comme" in p or "analogie" in p or "analogy" in p:
            return "analogy"
        # influence par policy (si reward social a augmenté la concrétude)
        try:
            pol = getattr(self.arch, "style_policy", None)
            concr = pol.params.get("concretude_bias", 0.6) if pol else 0.6
            return "deduction" if concr >= 0.55 else "abduction"
        except Exception:
            return "deduction"

    def _make_hypotheses(self, prompt: str, strategy: str) -> List[Hypothesis]:
        base = [
            Hypothesis(content="tu veux une explication avec étapes et tests", prior=0.55),
            Hypothesis(content="tu veux que j’auto-documente ce que j’apprends", prior=0.50),
            Hypothesis(content="tu veux un patch/exécution immédiate", prior=0.45),
        ]
        # léger ajustement par stratégie
        for h in base:
            if strategy == "abduction" and "expl" in h.content:
                h.prior += 0.05
            if strategy == "deduction" and "patch" in h.content:
                h.prior += 0.05
        return base

    def _score_hypotheses(self, prompt: str, hypos: List[Hypothesis], strategy: str) -> List[float]:
        scores = []
        for h in hypos:
            s = 0.4 * h.prior
            s += 0.2 * (1.0 if strategy in ("deduction", "analogy") else 0.0)
            s += 0.15 * (1.0 if "test" in h.content or "auto-documente" in h.content else 0.0)
            # style policy
            try:
                pol = getattr(self.arch, "style_policy", None)
                if pol:
                    s += 0.1 * pol.params.get("concretude_bias", 0.6)
                    s -= 0.05 * pol.params.get("hedging", 0.3)
            except Exception:
                pass
            scores.append(max(0.0, min(1.0, s)))
        return scores

    def _propose_tests(self, prompt: str, strategy: str) -> List[Test]:
        out: List[Test] = []
        if strategy == "deduction":
            out.append(Test(description="isoler la demande en 1 verbe + 1 objet et valider"))
            out.append(Test(description="proposer 2 chemins d’action et demander un choix"))
        elif strategy == "abduction":
            out.append(Test(description="formuler 2 hypothèses causales et demander un contre-exemple"))
            out.append(Test(description="sonder le contexte minimal (contrainte/ressource)"))
        else:  # analogy
            out.append(Test(description="trouver un cas similaire récent et transférer la solution"))
            out.append(Test(description="vérifier 1 différence critique entre les cas"))
        return out

    def _simulate_micro_inference(self, prompt: str, strategy: str, h: Hypothesis) -> str:
        # Ici tu peux brancher du code concret (parsing inbox, etc.)
        # Pour l’instant, on produit une evidence explicite et traçable:
        if strategy == "abduction":
            return f"Cause plausible: l’utilisateur veut {h.content}. Testons via contre-exemples."
        if strategy == "analogy":
            return f"Analogie: pattern similaire (précision demandée + prochain test)."
        return f"Plan déductif: valider intention → choisir test → exécuter micro-étape."

    def _estimate_complexity(self, prompt: str, strategy: str, hypos: List[Hypothesis]) -> float:
        L = max(10, len(prompt or ""))  # longueur
        base = 0.25 if strategy == "deduction" else (0.35 if strategy == "abduction" else 0.30)
        bump = 0.10 if L > 140 else 0.0
        return float(min(1.0, base + bump))

    def _push_episode(self, ep: Dict[str, Any]) -> None:
        self.reasoning_history["recent_inferences"].append(ep)
        st = self.reasoning_history["stats"]
        st["n_episodes"] += 1
        # préférences de stratégie (EWMA)
        strat = ep.get("strategy", "deduction")
        for k in list(st["strategy_preferences"].keys()):
            st["strategy_preferences"][k] *= 0.95
        st["strategy_preferences"][strat] = st["strategy_preferences"].get(strat, 0.0) + 0.05

    def _learn(self, final_conf: float, strategy: str) -> None:
        traj = self.reasoning_history["learning_trajectory"]
        traj.append({"t": time.time(), "confidence": float(final_conf), "strategy": strategy})

    def _make_readable_summary(
        self,
        strategy: str,
        hypos: List[Hypothesis],
        chosen_idx: int,
        tests: List[Test],
        evidence: Evidence,
        final_conf: float
    ) -> str:
        hyp = hypos[chosen_idx]
        test_desc = "; ".join(t.description for t in tests[:2]) if tests else "—"
        return (
            f"Stratégie {strategy} → hypothèse '{hyp.content}' avec {final_conf:.2f} conf."
            f" Tests: {test_desc}. Évidence: {evidence.notes}"
        )
