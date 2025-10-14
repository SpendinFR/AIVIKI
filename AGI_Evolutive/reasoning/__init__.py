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
            "learning_trajectory": deque(maxlen=200),   # [{ts, confidence}]
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
        self.reasoning_history["learning_trajectory"].append({
            "ts": rec["timestamp"],
            "confidence": float(final_confidence),
        })
