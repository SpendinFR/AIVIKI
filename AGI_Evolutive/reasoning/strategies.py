from typing import Any, Dict, List
import time
import re


class ReasoningStrategy:
    """Interface minimale pour une stratégie de raisonnement."""
    name: str = "base"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retourne un dict:
        {
          "notes": str,
          "proposals": [{"answer": str, "confidence": float, "support": List[str]}],
          "questions": [str],
          "cost": float,
          "time": float
        }
        """
        raise NotImplementedError(
            "ReasoningStrategy.apply doit être implémentée par les sous-classes"
        )


# ---------- 1) Décomposition (sous-problèmes) ----------
class DecompositionStrategy(ReasoningStrategy):
    name = "décomposition"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        questions: List[str] = []
        txt = (prompt or "").strip()

        # Heuristiques simples de sous-questions
        # - si "pourquoi/comment" → extraire causes/étapes
        # - sinon: extraire entités clés
        lower = txt.lower()
        if any(w in lower for w in ["pourquoi", "why"]):
            questions.append("Quelles sont les causes plausibles ?")
            questions.append("Quelles preuves observez-vous ?")
        if any(w in lower for w in ["comment", "how"]):
            questions.append("Quelles sont les étapes pour y parvenir ?")
            questions.append("Quels obstacles et hypothèses ?")

        # Mots clés -> sous-questions
        tokens = re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç0-9]+", lower)
        key = list(dict.fromkeys([t for t in tokens if len(t) > 4]))[:5]  # top 5 distincts
        if key:
            questions.append(f"Définir/clarifier: {', '.join(key[:3])}")

        notes = "Sous-problèmes identifiés" if questions else "Pas de sous-problèmes saillants"
        return {
            "notes": notes,
            "proposals": [],
            "questions": questions,
            "cost": 0.5,
            "time": time.time() - t0,
        }


# ---------- 2) Récupération d'évidence (mémoire/doc) ----------
class EvidenceRetrievalStrategy(ReasoningStrategy):
    name = "récupération"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        retrieve = toolkit.get("retrieve_fn")
        supports: List[str] = []

        if retrieve:
            # cherche 4 items (interactions/docs)
            hits = retrieve(prompt, top_k=4)
            for h in hits:
                title = h.get("meta", {}).get("title") or h.get("meta", {}).get("type", "")
                snippet = h.get("text", "")
                if len(snippet) > 220:
                    snippet = snippet[:220] + "…"
                label = f"{title}: {snippet}" if title else snippet
                supports.append(label)

        notes = "Évidence récupérée depuis la mémoire" if supports else "Aucune évidence en mémoire"
        return {
            "notes": notes,
            "proposals": [],  # pas de proposition finale ici, juste du support
            "questions": [],
            "cost": 1.0,
            "time": time.time() - t0,
            "support": supports,
        }


# ---------- 3) Génération / ranking d'hypothèses ----------
class HypothesisRankingStrategy(ReasoningStrategy):
    name = "hypothèses"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        support_snippets: List[str] = context.get("support", [])
        proposals: List[Dict[str, Any]] = []

        # Heuristique: si support, proposer une synthèse courte
        if support_snippets:
            joined = " | ".join(support_snippets[:3])
            ans = f"Synthèse appuyée sur mémoire: {joined}"
            conf = min(0.65 + 0.1 * min(len(support_snippets), 3), 0.9)
            proposals.append({"answer": ans, "confidence": conf, "support": support_snippets[:3]})

        # Proposer une réponse " prudente " quand info pauvre
        if not proposals:
            proposals.append({
                "answer": "Je manque d'évidence directe. Je propose d'éclaircir le but et les contraintes.",
                "confidence": 0.35,
                "support": []
            })

        return {
            "notes": "Hypothèses construites et pondérées",
            "proposals": proposals,
            "questions": [],
            "cost": 0.8,
            "time": time.time() - t0,
        }


# ---------- 4) Auto-vérification légère ----------
class SelfCheckStrategy(ReasoningStrategy):
    name = "auto-vérification"

    def apply(self, prompt: str, context: Dict[str, Any], toolkit: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        proposals: List[Dict[str, Any]] = context.get("proposals", [])
        last_answer = context.get("last_answer", "")
        notes = "Aucune contradiction apparente"

        # check basique: éviter contradictions " oui/non ", " vrai/faux "
        contradictions = [("oui", "non"), ("vrai", "faux"), ("possible", "impossible")]
        for p in proposals:
            a = (p.get("answer") or "").lower()
            for x, y in contradictions:
                if x in a and y in last_answer.lower():
                    p["confidence"] *= 0.8
                    notes = "Contradiction détectée avec l'itération précédente (pondération réduite)"
                if y in a and x in last_answer.lower():
                    p["confidence"] *= 0.8
                    notes = "Contradiction détectée avec l'itération précédente (pondération réduite)"

        # clamp
        for p in proposals:
            p["confidence"] = max(0.0, min(1.0, float(p["confidence"])))

        return {
            "notes": notes,
            "proposals": proposals,
            "questions": [],
            "cost": 0.2,
            "time": time.time() - t0,
        }
