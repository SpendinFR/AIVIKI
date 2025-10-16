"""Goal generation heuristics driven by curiosity signals."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class CuriosityEngine:
    """Generate candidate sub-goals based on simple heuristics.

    Cette version vit dans le module *goals* et s'occupe de traduire des
    signaux de curiosité (lacunes détectées, contradictions, concepts
    nouveaux) en objectifs concrets pour le :class:`DagStore`.  Elle est
    complémentaire au moteur de curiosité du package ``learning`` qui ne
    produit qu'une récompense intrinsèque numérique.
    """

    def __init__(self, architecture=None):
        self.architecture = architecture

    def suggest_subgoals(
        self,
        parent_goal: Optional[Dict[str, Any]] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` sub-goal dictionaries for :class:`DagStore`."""

        context = self._collect_context()
        gaps = self._identify_gaps(context)
        proposals = [self._gap_to_goal(gap, parent_goal) for gap in gaps]

        if len(proposals) < k:
            proposals.append(
                {
                    "description": "Explorer un concept peu maîtrisé et produire une synthèse.",
                    "criteria": ["Fournir un résumé structuré et une auto-évaluation."],
                    "created_by": "curiosity",
                    "value": 0.55,
                    "competence": 0.5,
                    "curiosity": 0.8,
                    "urgency": 0.35,
                    "parent_ids": [parent_goal["id"]] if parent_goal else [],
                }
            )

        random.shuffle(proposals)
        return proposals[:k]

    # ------------------------------------------------------------------
    def _collect_context(self) -> Dict[str, Any]:
        metacog = getattr(self.architecture, "metacognition", None)
        reasoning = getattr(self.architecture, "reasoning", None)
        memory = getattr(self.architecture, "memory", None)

        status: Dict[str, Any] = {}
        if metacog and hasattr(metacog, "get_metacognitive_status"):
            try:
                status = metacog.get_metacognitive_status()
            except Exception:
                status = {}

        reasoning_stats: Dict[str, Any] = {}
        if reasoning and hasattr(reasoning, "get_reasoning_stats"):
            try:
                reasoning_stats = reasoning.get_reasoning_stats()
            except Exception:
                reasoning_stats = {}

        # Concepts récemment extraits mais pas encore “appris”
        novel_concepts: List[str] = []
        known_concepts: set = set()
        try:
            if memory and hasattr(memory, "get_recent_memories"):
                recents = memory.get_recent_memories(200)
                # connus (notes / appris)
                for item in recents:
                    kind = (item.get("kind") or item.get("type") or "").lower()
                    if kind in {"concept_note", "concept_learned"}:
                        c = item.get("concept") or (item.get("metadata", {}) or {}).get("concept")
                        if c:
                            known_concepts.add(str(c).strip())
                        for c, _score in (item.get("metadata", {}) or {}).get("concepts", []):
                            known_concepts.add(str(c).strip())
                # nouveaux (extrait mais pas appris)
                for item in recents:
                    kind = (item.get("kind") or item.get("type") or "").lower()
                    if kind == "concept_extracted":
                        meta = (item.get("metadata") or {})
                        concepts = meta.get("concepts") or []
                        for entry in concepts:
                            c = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                            c = str(c).strip()
                            if c and c not in known_concepts and c not in novel_concepts:
                                novel_concepts.append(c)
        except Exception:
            pass

        contradictions: List[Dict[str, Any]] = []
        beliefs = getattr(self.architecture, "beliefs", None)
        if beliefs:
            try:
                for positive, negative in beliefs.find_contradictions(min_conf=0.7):
                    contradictions.append(
                        {
                            "subject": positive.subject,
                            "relation": positive.relation,
                            "value": positive.value,
                            "positive": positive.id,
                            "negative": negative.id,
                        }
                    )
            except Exception:
                contradictions = []

        return {
            "metacog": status,
            "reasoning": reasoning_stats,
            "novel_concepts": novel_concepts,
            "contradictions": contradictions,
        }

    def _identify_gaps(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        performance = context.get("metacog", {}).get("performance_metrics", {})
        low_metrics = sorted(
            (item for item in performance.items() if item[1] < 0.45),
            key=lambda item: item[1],
        )

        gaps: List[Dict[str, Any]] = [
            {"domain": name, "score": value, "severity": float(1.0 - value)}
            for name, value in low_metrics[:3]
        ]

        reasoning_errors = context.get("reasoning", {}).get("common_errors", [])
        gaps.extend(
            {"domain": "reasoning_error", "score": 0.3, "hint": err, "severity": 0.6}
            for err in reasoning_errors[:2]
        )

        # NOUVEAU : apprendre des concepts nouveaux (générique)
        for c in context.get("novel_concepts", [])[:3]:
            gaps.append({"domain": "novel_concept", "concept": c, "score": 0.4, "severity": 0.6})

        for contradiction in context.get("contradictions", [])[:1]:
            gaps.append(
                {
                    "domain": "belief_contradiction",
                    "subject": contradiction.get("subject"),
                    "relation": contradiction.get("relation"),
                    "positive": contradiction.get("positive"),
                    "negative": contradiction.get("negative"),
                    "severity": 0.85,
                }
            )

        if not gaps:
            gaps.append({"domain": "exploration", "score": 0.5, "severity": 0.4})

        return gaps

    def _gap_to_goal(self, gap: Dict[str, Any], parent_goal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        description = self._describe_gap(gap)
        criteria = self._default_criteria(gap)
        parent_ids = [parent_goal["id"]] if parent_goal and "id" in parent_goal else []

        base_value = 0.5 + 0.2 * random.random()
        competence = max(0.3, min(0.7, 0.45 + (random.random() - 0.5) * 0.3))
        curiosity = min(1.0, 0.6 + 0.3 * random.random())
        urgency = 0.3 + 0.2 * random.random()

        # Cas générique : concept nouveau → objectif d’apprentissage “naturel”
        if gap.get("domain") == "novel_concept" and gap.get("concept"):
            c = str(gap["concept"]).strip()
            description = f"Apprendre le concept « {c} » et produire une synthèse exploitable."
            criteria = [
                f"Définir « {c} » en 3 phrases maximum.",
                "Donner 2 exemples et 1 contre-exemple pertinents.",
                "Énoncer 1 règle de décision utilisant ce concept.",
            ]
        if gap.get("domain") == "belief_contradiction":
            subj = str(gap.get("subject", "?")).strip()
            rel = str(gap.get("relation", "?")).strip()
            description = f"Résoudre contradiction « {subj}, {rel} » dans le graphe de croyances."
            criteria = [
                "Lister les preuves soutenant chaque version.",
                "Identifier une observation décisive à collecter.",
                "Mettre à jour la croyance avec justification." ,
            ]

        return {
            "description": description,
            "criteria": criteria,
            "created_by": "curiosity",
            "value": base_value,
            "competence": competence,
            "curiosity": curiosity,
            "urgency": urgency,
            "parent_ids": parent_ids,
        }

    def _describe_gap(self, gap: Dict[str, Any]) -> str:
        domain = gap.get("domain")
        if domain == "reasoning_error":
            return f"Analyser et corriger une erreur de raisonnement: {gap.get('hint', 'non spécifié')}"
        if domain == "exploration":
            return "Explorer un nouveau sujet pour enrichir la base de connaissances."
        if domain == "novel_concept":
            return "Apprendre un concept récemment rencontré."
        if domain == "belief_contradiction":
            subj = gap.get("subject", "?")
            rel = gap.get("relation", "?")
            return f"Résoudre contradiction « {subj}, {rel} » identifiée dans le graphe."
        return f"Améliorer la métrique {domain} par une expérimentation ciblée."

    def _default_criteria(self, gap: Dict[str, Any]) -> List[str]:
        domain = gap.get("domain")
        if domain == "reasoning_error":
            return ["Documenter 3 contre-exemples et une stratégie de prévention."]
        if domain == "exploration":
            return ["Produire une carte mentale du sujet exploré."]
        if domain == "belief_contradiction":
            return ["Comparer les justifications et collecter une preuve supplémentaire."]
        return ["Mesurer une amélioration significative après 3 essais contrôlés."]
