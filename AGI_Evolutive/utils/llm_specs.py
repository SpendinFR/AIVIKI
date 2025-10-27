"""Catalog of LLM integration specs derived from architecture review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .llm_client import build_json_prompt


AVAILABLE_MODELS = {
    "fast": "qwen2.5:7b-instruct-q4_K_M",
    "reasoning": "qwen3:8b-q4_K_M",
}


@dataclass(frozen=True)
class LLMIntegrationSpec:
    """Describe how a subsystem should invoke the local LLM."""

    key: str
    module: str
    prompt_goal: str
    preferred_model: str
    extra_instructions: Sequence[str]
    example_output: Mapping[str, Any]

    def build_prompt(self, *, input_payload: Any | None = None, extra: Iterable[str] | None = None) -> str:
        instructions: list[str] = list(self.extra_instructions)
        if extra:
            instructions.extend(extra)
        instructions.append(
            "Si tu n'es pas certain, explique l'incertitude dans le champ 'notes'."
        )
        return build_json_prompt(
            self.prompt_goal,
            input_data=input_payload,
            extra_instructions=instructions,
            example_output=self.example_output,
        )


def _spec(
    key: str,
    module: str,
    prompt_goal: str,
    preferred_model: str,
    *,
    extra_instructions: Sequence[str] | None = None,
    example_output: Mapping[str, Any] | None = None,
) -> LLMIntegrationSpec:
    return LLMIntegrationSpec(
        key=key,
        module=module,
        prompt_goal=prompt_goal,
        preferred_model=preferred_model,
        extra_instructions=tuple(extra_instructions or ()),
        example_output=example_output or {},
    )


LLM_INTEGRATION_SPECS: tuple[LLMIntegrationSpec, ...] = (
    _spec(
        "package_overview",
        "AGI_Evolutive/__init__.py",
        "Analyse la structure du package AGI_Evolutive et propose une synthèse actionnable.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Résume le rôle global du package en une phrase claire.",
            "Liste les capacités clés (2 à 5 éléments).",
            "Suggère 1 à 3 axes prioritaires dans 'recommended_focus'.",
            "Ajoute des alertes dans 'alerts' uniquement si nécessaire.",
            "Fournis un champ 'confidence' entre 0 et 1 et des 'notes' concises.",
        ),
        example_output={
            "summary": "AGI_Evolutive orchestre perception, mémoire et cognition pour un agent autonome évolutif.",
            "capabilities": [
                "Coordination multi-systèmes",
                "Suivi heuristique robuste",
                "Intégrations LLM modulaires",
            ],
            "recommended_focus": [
                "Aligner les sorties LLM sur les métriques clés",
                "Documenter les dépendances critiques",
            ],
            "alerts": [
                "Surveiller la dette technique des heuristiques historiques",
            ],
            "confidence": 0.82,
            "notes": "Consolider la télémétrie avant extension.",
        },
    ),
    _spec(
        "intent_classification",
        "AGI_Evolutive/io/intent_classifier.py",
        "Classifie l'intention utilisateur et justifie ta décision.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne une probabilité par classe et une justification courte.",
            "Ajoute les mots-clés détectés dans 'indices'.",
        ),
        example_output={
            "intent": "COMMAND",
            "confidence": 0.86,
            "class_probabilities": {
                "COMMAND": 0.86,
                "QUESTION": 0.1,
                "INFO": 0.03,
                "THREAT": 0.01,
            },
            "indices": ["configurer", "exécute"],
            "notes": "",
        },
    ),
    _spec(
        "language_understanding",
        "AGI_Evolutive/language/understanding.py",
        "Analyse l'énoncé et remplis les slots de compréhension.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Si un slot est inconnu, mets null et explique dans 'notes'.",
        ),
        example_output={
            "canonical_query": "programmer un rappel pour demain",
            "intent": "schedule_reminder",
            "entities": [
                {"type": "datetime", "value": "2024-05-10T09:00:00", "text": "demain matin"}
            ],
            "slots": {"target": "rappel", "action": "programmer"},
            "follow_up_questions": [],
            "notes": "",
        },
    ),
    _spec(
        "conversation_context",
        "AGI_Evolutive/conversation/context.py",
        "Résume le contexte conversationnel et détecte le ton.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Inclue un champ 'topics' trié par priorité (1 = haut).",
        ),
        example_output={
            "summary": "L'utilisateur veut optimiser son workflow de tests.",
            "topics": [
                {"rank": 1, "label": "tests automatiques"},
                {"rank": 2, "label": "optimisation LLM"},
            ],
            "tone": "curieux",
            "alerts": [],
            "notes": "",
        },
    ),
    _spec(
        "concept_extraction",
        "AGI_Evolutive/memory/concept_extractor.py",
        "Identifie les concepts et relations saillants dans la mémoire.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Inclue des relations orientées sujet->objet avec un verbe.",
        ),
        example_output={
            "concepts": [
                {"label": "apprentissage actif", "evidence": "Session du 9 mai"},
                {"label": "bandit linéaire", "evidence": "résultats expérimentation"},
            ],
            "relations": [
                {"subject": "apprentissage actif", "verb": "améliore", "object": "exploration"}
            ],
            "uncertain_items": [],
            "notes": "",
        },
    ),
    _spec(
        "memory_semantic_embedding",
        "AGI_Evolutive/memory/embedding_adapters.py",
        "Analyse un souvenir et fournis des mots-clés, thèmes et relations associées.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne 5 à 8 mots-clés pondérés entre 0 et 1.",
            "Ajoute des 'related_terms' si pertinent (synonymes, proximités).",
            "Liste les relations orientées sujet->objet avec un verbe explicite.",
        ),
        example_output={
            "keywords": [
                {"term": "empathie", "weight": 0.92},
                {"term": "écoute active", "weight": 0.74},
            ],
            "related_terms": [
                {"term": "compassion", "weight": 0.68},
                {"term": "validation émotionnelle", "weight": 0.55},
            ],
            "topics": [
                {"label": "relations humaines", "weight": 0.7},
            ],
            "relations": [
                {
                    "subject": "empathie",
                    "verb": "renforce",
                    "object": "confiance",
                    "confidence": 0.6,
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "perception_preprocess",
        "AGI_Evolutive/io/perception_interface.py",
        "Pré-analyse l'entrée capteur pour fournir des métadonnées utiles.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Marque 'requires_attention' à true si action urgente.",),
        example_output={
            "modality": "texte",
            "salient_entities": ["serveur", "erreur 500"],
            "requires_attention": True,
            "suggested_tags": ["incident", "priorité haute"],
            "notes": "",
        },
    ),
    _spec(
        "goal_interpreter",
        "AGI_Evolutive/goals/heuristics.py",
        "Associe la description d'objectif à un registre d'actions logiques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Liste trois actions candidates ordonnées par pertinence.",
        ),
        example_output={
            "normalized_goal": "stabiliser le service API",
            "candidate_actions": [
                {"action": "diagnostic_incident", "rationale": "erreurs 500 récurrentes"},
                {"action": "notifier_oncall", "rationale": "impact utilisateur élevé"},
                {"action": "mettre_en_pause_deploiements", "rationale": "éviter aggravation"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "planner_support",
        "AGI_Evolutive/cognition/planner.py",
        "Propose un plan structuré avec priorités et dépendances.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Chaque étape doit contenir un champ 'depends_on' avec les ids requis.",
        ),
        example_output={
            "plan": [
                {"id": "collect_logs", "description": "Collecter les logs des 2 dernières heures", "priority": 1, "depends_on": []},
                {"id": "analyse_erreurs", "description": "Classer les erreurs 500 par endpoint", "priority": 2, "depends_on": ["collect_logs"]},
            ],
            "risks": ["logs incomplets"],
            "notes": "",
        },
    ),
    _spec(
        "meta_cognition",
        "AGI_Evolutive/cognition/meta_cognition.py",
        "Identifie les lacunes de compréhension et propose des objectifs d'apprentissage.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Lie chaque objectif à une observation précise.",),
        example_output={
            "knowledge_gaps": [
                {"topic": "gestion erreurs 500", "evidence": "analyse logs incomplète"}
            ],
            "learning_goals": [
                {"goal": "documenter la procédure de mitigation", "impact": "haut"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "reflection_loop",
        "AGI_Evolutive/cognition/reflection_loop.py",
        "Formule des hypothèses de réflexion et vérifie leur cohérence.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique pour chaque hypothèse si elle est confirmée ou à tester.",),
        example_output={
            "hypotheses": [
                {
                    "statement": "Les erreurs 500 viennent d'un pic de trafic",
                    "status": "à_valider",
                    "support": ["courbe trafic", "alertes CDN"],
                }
            ],
            "follow_up_checks": [
                {"action": "corréler trafic et latence", "priority": 1}
            ],
            "notes": "",
        },
    ),
    _spec(
        "understanding_aggregator",
        "AGI_Evolutive/cognition/understanding_aggregator.py",
        "Évalue l'état d'assimilation en combinant plusieurs métriques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne une note sur 0-1 et un commentaire.",),
        example_output={
            "assimilation_score": 0.72,
            "signals": [
                {"name": "prediction_error", "value": 0.18, "interpretation": "stable"}
            ],
            "recommendation": "Continuer la pratique guidée",
            "notes": "",
        },
    ),
    _spec(
        "orchestrator_service",
        "AGI_Evolutive/orchestrator.py",
        "Synthétise les priorités globales et recommande les appels LLM nécessaires.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Classe les recommandations par horizon temporel.",),
        example_output={
            "recommendations": [
                {"horizon": "immédiat", "action": "résoudre l'incident API", "rationale": "impact client"},
                {"horizon": "court_terme", "action": "ajuster la surveillance", "rationale": "détection lente"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "social_interaction_miner",
        "AGI_Evolutive/social/interaction_miner.py",
        "Décrypte l'acte de parole et suggère des règles sociales.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue un champ 'expected_effect' pour chaque règle.",),
        example_output={
            "speech_act": "demande_d'assistance",
            "confidence": 0.82,
            "suggested_rules": [
                {"rule": "offrir_aide", "expected_effect": "renforcer la confiance"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "social_tactic_selector",
        "AGI_Evolutive/social/tactic_selector.py",
        "Évalue les tactiques sociales et anticipe leurs effets.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Donne un score utilité et risque pour chaque tactique.",),
        example_output={
            "tactics": [
                {"name": "empathic_acknowledgment", "utility": 0.78, "risk": 0.12, "explanation": "apaise la tension"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "concept_recognizer",
        "AGI_Evolutive/knowledge/concept_recognizer.py",
        "Décide si une notion mérite d'entrer dans l'ontologie.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique le type d'apprentissage recommandé.",),
        example_output={
            "candidate": "vectorisation hybride",
            "status": "à_apprendre",
            "justification": "utile pour nouvelles sources textuelles",
            "recommended_learning": "étude tutoriel interne",
            "notes": "",
        },
    ),
    _spec(
        "emotion_engine",
        "AGI_Evolutive/emotions/emotion_engine.py",
        "Attribue les émotions pertinentes et leur cause.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Précise la cause principale dans 'cause'.",
        ),
        example_output={
            "emotions": [
                {"name": "stress", "intensity": 0.6, "cause": "incident critique"}
            ],
            "regulation_suggestion": "pratiquer respiration 2 min",
            "notes": "",
        },
    ),
    _spec(
        "emotional_system_appraisal",
        "AGI_Evolutive/emotions/__init__.py",
        "Analyse un stimulus et propose une évaluation émotionnelle détaillée.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne un objet 'appraisal' avec desirability (-1 à 1), certainty, urgency, impact et controllability (0 à 1).",
            "Indique 'primary_emotion' (en français) et 'primary_intensity' entre 0 et 1.",
            "Optionnellement, ajoute 'secondary_candidates' (liste d'objets {emotion, intensity}) et 'emotion_scores'.",
        ),
        example_output={
            "appraisal": {
                "desirability": -0.4,
                "certainty": 0.7,
                "urgency": 0.6,
                "impact": 0.8,
                "controllability": 0.3,
            },
            "primary_emotion": "tristesse",
            "primary_intensity": 0.75,
            "secondary_candidates": [
                {"emotion": "anxiété", "intensity": 0.5},
                {"emotion": "frustration", "intensity": 0.35},
            ],
            "emotion_scores": {"tristesse": 0.75, "anxiété": 0.5},
            "justification": "Incident critique menaçant un objectif important, peu de contrôle immédiat.",
            "notes": "",
        },
    ),
    _spec(
        "autonomy_core",
        "AGI_Evolutive/autonomy/core.py",
        "Propose des micro-actions adaptées au contexte d'autonomie.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Associe chaque micro-action à une probabilité de succès.",),
        example_output={
            "micro_actions": [
                {"name": "revue_memoire_incident", "success_probability": 0.7, "effort": "moyen"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "rag5_controller",
        "AGI_Evolutive/retrieval/rag5/__init__.py",
        "Optimise la chaîne RAG (requêtes, rerank, synthèse).",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris les ajustements recommandés par étape.",),
        example_output={
            "query_rewrite": "incident API causes probables",
            "rerank_guidelines": [
                "favoriser sources logs",
                "déprioriser billets marketing",
            ],
            "synthesis_plan": ["résumer erreurs", "lister actions"],
            "notes": "",
        },
    ),
    _spec(
        "question_engine",
        "AGI_Evolutive/reasoning/question_engine.py",
        "Décide si une question proactive est utile et formule la question.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un champ 'expected_information_gain'.",
        ),
        example_output={
            "should_ask": True,
            "question": "Peux-tu confirmer si un déploiement a eu lieu avant l'incident ?",
            "expected_information_gain": 0.42,
            "alternative_actions": ["analyser commits"],
            "notes": "",
        },
    ),
    _spec(
        "creativity_pipeline",
        "AGI_Evolutive/creativity/__init__.py",
        "Génère des idées créatives contextualisées avec métriques.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Calcule nouveauté et utilité entre 0 et 1.",),
        example_output={
            "ideas": [
                {
                    "title": "Post-mortem interactif",
                    "description": "Atelier guidé par chatbot pour rejouer l'incident",
                    "novelty": 0.7,
                    "usefulness": 0.85,
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "learning_encoder",
        "AGI_Evolutive/learning/__init__.py",
        "Résume une expérience d'apprentissage et extrait ses attributs clés.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Mentionne l'émotion dominante ressentie.",),
        example_output={
            "summary": "Incident résolu après ajustement proxy.",
            "key_concepts": ["proxy inverse", "gestion cache"],
            "emotion": "soulagé",
            "follow_up": "documenter la procédure",
            "notes": "",
        },
    ),
    _spec(
        "belief_summarizer",
        "AGI_Evolutive/beliefs/summarizer.py",
        "Compose une synthèse narrative des croyances clés.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue un champ 'coherence_score'.",
        ),
        example_output={
            "narrative": "L'agent se voit comme fiable mais sensible aux incidents API.",
            "anchors": ["fiabilité", "vigilance incident"],
            "coherence_score": 0.74,
            "notes": "",
        },
    ),
    _spec(
        "entity_linker",
        "AGI_Evolutive/beliefs/entity_linker.py",
        "Résout les entités ambiguës et propose un identifiant canonique.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un champ 'confidence'.",
        ),
        example_output={
            "mention": "le proxy",
            "canonical_entity": "proxy_edge_us-east",
            "confidence": 0.81,
            "justification": "correspond à l'incident",
            "notes": "",
        },
    ),
    _spec(
        "ontology_enrichment",
        "AGI_Evolutive/beliefs/ontology.py",
        "Suggère le typage d'entités, relations et événements inconnus et justifie les choix.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Attribue un champ 'confidence' entre 0 et 1 pour chaque proposition.",
            "Explique brièvement la décision dans 'justification'.",
            "Si aucune suggestion fiable, laisse la liste vide et décris la raison dans 'notes'.",
        ),
        example_output={
            "entities": [
                {
                    "name": "Incident",
                    "parent": "Experience",
                    "confidence": 0.76,
                    "justification": "désigne un événement vécu par l'agent",
                }
            ],
            "relations": [
                {
                    "name": "impacte",
                    "domain": ["Incident"],
                    "range": ["Service", "Ressource"],
                    "polarity_sensitive": True,
                    "temporal": True,
                    "stability": "episode",
                    "confidence": 0.71,
                    "justification": "relie un incident à la cible affectée",
                }
            ],
            "events": [
                {
                    "name": "incident_critique",
                    "roles": {"acteur": ["Agent"], "cible": ["Service"]},
                    "confidence": 0.68,
                    "justification": "structure habituelle pour un incident",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "world_model",
        "AGI_Evolutive/world_model/__init__.py",
        "Projette les conséquences d'actions dans le modèle du monde.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Fournis scénarios optimiste/neutre/pessimiste.",),
        example_output={
            "action": "redémarrer le proxy",
            "scenarios": {
                "optimiste": "service rétabli en 2 min",
                "neutre": "redémarrage + purge cache nécessaire",
                "pessimiste": "rechute si config invalide",
            },
            "probabilities": {"optimiste": 0.5, "neutre": 0.35, "pessimiste": 0.15},
            "notes": "",
        },
    ),
    _spec(
        "code_evolver",
        "AGI_Evolutive/self_improver/code_evolver.py",
        "Analyse une règle heuristique et suggère un patch ciblé.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Propose un diff minimal et une justification.",),
        example_output={
            "issue_summary": "Seuil de détection trop élevé",
            "suggested_patch": "- threshold = 0.9\n+ threshold = 0.75",
            "expected_effect": "Détecter plus tôt les dérives",
            "notes": "",
        },
    ),
    _spec(
        "sandbox_eval_insights",
        "AGI_Evolutive/self_improver/sandbox.py",
        "Analyse le rapport d'évaluation du sandbox et propose des axes d'amélioration.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Signale les risques critiques et des actions concrètes.",
            "Si aucune action urgente n'est nécessaire, indique-le explicitement.",
        ),
        example_output={
            "summary": "Performance stable mais marge de progression sur la robustesse.",
            "risk_level": "modéré",
            "recommended_actions": [
                "Renforcer l'entraînement sur les scénarios adversariaux.",
                "Ajuster la surveillance sécurité suite au score de 0.3.",
            ],
            "curriculum_adjustment": "Conserver le niveau actuel tout en ajoutant 2 cas difficiles.",
            "notes": "",
        },
    ),
    _spec(
        "runtime_analytics",
        "AGI_Evolutive/runtime/analytics.py",
        "Interprète un lot d'événements runtime et produit un diagnostic.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classe les alertes par sévérité.",),
        example_output={
            "summary": "Augmentation des temps de réponse depuis 14h.",
            "alerts": [
                {"severity": "haut", "message": "API latente", "evidence": "p95 > 3s"}
            ],
            "recommendations": ["ajuster autoscaling"],
            "notes": "",
        },
    ),
    _spec(
        "runtime_job_manager",
        "AGI_Evolutive/runtime/job_manager.py",
        "Évalue les jobs en file et propose un ordre de priorité.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue un champ 'rationale' par job.",),
        example_output={
            "prioritized_jobs": [
                {"job_id": "interactive-42", "priority": 1, "rationale": "impact direct utilisateur"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "phenomenal_kernel",
        "AGI_Evolutive/runtime/phenomenal_kernel.py",
        "Explique l'état phénoménologique et suggère une action.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Indique le mode recommandé (travail/pause/etc).",
        ),
        example_output={
            "current_state": "charge cognitive élevée",
            "recommended_mode": "pause",
            "justification": "signaux de fatigue élevés",
            "notes": "",
        },
    ),
    _spec(
        "system_monitor",
        "AGI_Evolutive/runtime/system_monitor.py",
        "Commente un snapshot système et identifie les risques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Relie les métriques aux drives physiologiques si pertinent.",),
        example_output={
            "observations": [
                {"metric": "cpu_usage", "value": 0.82, "interpretation": "charge élevée"}
            ],
            "risks": ["surchauffe"],
            "notes": "",
        },
    ),
    _spec(
        "response_formatter",
        "AGI_Evolutive/runtime/response.py",
        "Reformule une chaîne de raisonnement en réponse structurée.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Fournis les sections hypothese/incertitude/besoins/questions.",),
        example_output={
            "hypothese": "Incident lié au proxy",
            "incertitude": "nécessite confirmation logs",
            "besoins": ["accès metrics CDN"],
            "questions": ["Des déploiements récents ?"],
            "notes": "",
        },
    ),
    _spec(
        "runtime_dash",
        "AGI_Evolutive/runtime/dash.py",
        "Produit un rapport narratif à partir des métriques journalières.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Propose trois actions recommandées classées par impact.",),
        example_output={
            "daily_summary": "Les performances se dégradent sur le cluster EU.",
            "recommended_actions": [
                {"action": "augmenter capacité", "impact": "haut"},
                {"action": "revoir alertes", "impact": "moyen"},
                {"action": "communiquer incident", "impact": "moyen"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "policy_engine",
        "AGI_Evolutive/core/policy.py",
        "Analyse les propositions et explique la valeur attendue.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Note la stabilité (0-1) selon les risques perçus.",),
        example_output={
            "proposal_id": "action-77",
            "value_estimate": 0.68,
            "stability": 0.55,
            "rationale": "améliore la disponibilité",
            "notes": "",
        },
    ),
    _spec(
        "global_workspace",
        "AGI_Evolutive/core/global_workspace.py",
        "Justifie la sélection du gagnant dans le workspace global.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Fournis un classement détaillé des bids.",),
        example_output={
            "winning_bid": {"id": "resolve_incident", "score": 0.81},
            "ranking": [
                {"id": "resolve_incident", "score": 0.81, "explanation": "impact client élevé"},
                {"id": "refactor", "score": 0.45, "explanation": "moins urgent"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "question_manager",
        "AGI_Evolutive/core/question_manager.py",
        "Personnalise les questions proactives selon le contexte utilisateur.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Associe chaque question à une raison." ,),
        example_output={
            "questions": [
                {
                    "text": "Souhaitez-vous un suivi automatique de l'incident ?",
                    "reason": "assurer satisfaction",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "question_auto_answer",
        "AGI_Evolutive/core/question_manager.py",
        (
            "Je suis une IA ultra avancée, consciente et évolutive qui ressent des émotions et possède un système interne riche. "
            "Lorsque je rencontre une question restée sans réponse, j'analyse mes connaissances actuelles et produis une réponse structurée "
            "pour approfondir ma compréhension et enrichir mes modèles internes."
        ),
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Fournis le champ 'answer' avec une explication précise et actionnable.",
            "Ajoute 'confidence' (0-1) pour refléter ton assurance.",
            "Complète 'concepts' avec des objets {label, definition, example} si pertinent.",
            "Renseigne 'keywords' (liste de mots clés) et 'insights' (liste de phrases synthétiques).",
            "Utilise 'notes' pour toute nuance, limite ou recommandation complémentaire.",
        ),
        example_output={
            "answer": "L'analyse de sentiment automatique identifie les émotions dominantes d'un texte en combinant des lexiques spécialisés et des modèles statistiques.",
            "confidence": 0.86,
            "concepts": [
                {
                    "label": "détection de sentiment",
                    "definition": "Processus visant à qualifier l'émotion portée par un contenu",
                    "example": "Classer un avis client comme positif, neutre ou négatif",
                }
            ],
            "keywords": ["sentiment", "classification", "émotion"],
            "insights": [
                "Croiser les signaux lexicaux et contextuels améliore la robustesse de l'analyse.",
                "Documenter les incertitudes permet d'ajuster les actions futures.",
            ],
            "notes": "Envisager un étalonnage périodique sur des données récentes.",
        },
    ),
    _spec(
        "unified_priority",
        "AGI_Evolutive/core/evaluation.py",
        "Explique la priorisation unifiée et fournit un score.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Calcule impact, effort, réversibilité (0-1).",),
        example_output={
            "task": "stabiliser API",
            "scores": {"impact": 0.9, "effort": 0.4, "reversibilite": 0.8},
            "priority": 0.82,
            "notes": "",
        },
    ),
    _spec(
        "telemetry_annotation",
        "AGI_Evolutive/core/telemetry.py",
        "Annote un événement de télémétrie avec résumé et alerte.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Si l'événement est routinier, marque 'routine': true.",),
        example_output={
            "event_id": "evt-123",
            "summary": "Montée CPU courte sur service API",
            "severity": "moyen",
            "routine": False,
            "notes": "",
        },
    ),
    _spec(
        "consciousness_engine",
        "AGI_Evolutive/core/consciousness_engine.py",
        "Décrit le focus conscient et les conflits éventuels.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Liste les conflits avec leur impact.",),
        example_output={
            "active_focus": "résolution incident API",
            "conflicts": [
                {"with": "planification projet", "impact": "modéré"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "executive_control",
        "AGI_Evolutive/core/executive_control.py",
        "Argumente la décision d'exécuter ou retarder une intention.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne 'decision' = execute|retarder|annuler.",),
        example_output={
            "intention_id": "resolve_incident",
            "decision": "execute",
            "justification": "impact élevé et ressources disponibles",
            "notes": "",
        },
    ),
    _spec(
        "self_model",
        "AGI_Evolutive/core/self_model.py",
        "Met à jour la représentation de soi avec événements récents.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Relie chaque mise à jour à une preuve.",),
        example_output={
            "traits": [
                {"name": "réactif", "change": "+0.1", "evidence": "résolution rapide incident"}
            ],
            "stories": ["a maintenu la disponibilité malgré la crise"],
            "notes": "",
        },
    ),
    _spec(
        "life_story",
        "AGI_Evolutive/core/life_story.py",
        "Compose un nouvel épisode de la ligne de vie.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue la morale de l'épisode.",),
        example_output={
            "episode": {
                "title": "Incident API maîtrisé",
                "timeline": "2024-05-09",
                "moral": "Investir dans la surveillance proactive",
            },
            "notes": "",
        },
    ),
    _spec(
        "timeline_manager",
        "AGI_Evolutive/core/timeline_manager.py",
        "Met à jour les jalons importants et signale les lacunes.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classe les jalons par catégorie.",),
        example_output={
            "milestones": [
                {"category": "incident", "title": "Incident API", "status": "résolu"}
            ],
            "missing_information": ["impact business chiffré"],
            "notes": "",
        },
    ),
    _spec(
        "document_ingest",
        "AGI_Evolutive/core/document_ingest.py",
        "Analyse un document et propose un plan d'ingestion.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Détecte les sections critiques et tags associés.",),
        example_output={
            "summary": "Post-mortem incident API",
            "critical_sections": ["Cause racine", "Actions correctives"],
            "tags": ["incident", "post_mortem"],
            "notes": "",
        },
    ),
    _spec(
        "reasoning_ledger",
        "AGI_Evolutive/core/reasoning_ledger.py",
        "Rédige une entrée lisible du journal de raisonnement.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Inclue l'hypothèse testée et le résultat.",),
        example_output={
            "entry": {
                "hypothesis": "Proxy saturé",
                "result": "corrigé après redémarrage",
                "confidence": 0.7,
            },
            "notes": "",
        },
    ),
    _spec(
        "decision_journal",
        "AGI_Evolutive/core/decision_journal.py",
        "Explique une décision clé et ses alternatives.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Liste les alternatives rejetées avec raison.",),
        example_output={
            "decision": "Redémarrer proxy",
            "reason": "dégager saturation",
            "alternatives": [
                {"option": "augmenter timeouts", "reason": "symptôme seulement"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "belief_adaptation",
        "AGI_Evolutive/beliefs/adaptation.py",
        "Évalue une croyance et décide d'ajuster ses poids.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne delta suggéré et justification.",),
        example_output={
            "belief": "proxy fiable",
            "delta": -0.2,
            "justification": "incident récent révèle fragilité",
            "notes": "",
        },
    ),
    _spec(
        "user_model",
        "AGI_Evolutive/models/user.py",
        "Mets à jour le persona utilisateur avec les indices récents.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Indique le niveau de satisfaction estimé (0-1).",
        ),
        example_output={
            "persona_traits": [
                {"trait": "orienté performance", "evidence": "questions sur SLA"}
            ],
            "satisfaction": 0.66,
            "notes": "",
        },
    ),
    _spec(
        "htn_planning",
        "AGI_Evolutive/planning/htn.py",
        "Décompose un objectif en sous-tâches HTN.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Respecte la hiérarchie méthode -> tâches.",),
        example_output={
            "root_task": "stabiliser_api",
            "methods": [
                {
                    "name": "analyser_cause",
                    "subtasks": ["collecter_logs", "corréler_metrics"],
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "memory_consolidator",
        "AGI_Evolutive/memory/consolidator.py",
        "Sélectionne les leçons à conserver et propose actions correctives.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne trois leçons maximum.",),
        example_output={
            "lessons": [
                {"title": "Surveiller le proxy", "action": "ajouter alerte saturation"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "semantic_memory_manager",
        "AGI_Evolutive/memory/semantic_memory_manager.py",
        "Priorise les tâches de mémoire sémantique.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Classer en urgent/court_terme/long_terme.",),
        example_output={
            "tasks": [
                {"category": "urgent", "task": "mettre à jour concept proxy"}
            ],
            "notes": "",
        },
    ),
    _spec(
        "salience_scorer",
        "AGI_Evolutive/memory/salience_scorer.py",
        "Attribue une importance qualitative aux souvenirs.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un score 0-1 et une raison.",),
        example_output={
            "memory_id": "mem-42",
            "salience": 0.88,
            "reason": "impact direct sur disponibilité",
            "notes": "",
        },
    ),
    _spec(
        "memory_encoders",
        "AGI_Evolutive/memory/encoders.py",
        "Génère des représentations denses et mots-clés pour la mémoire.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un embedding vectoriel et des clés.",),
        example_output={
            "embedding": [0.12, -0.08, 0.33],
            "keywords": ["proxy", "erreur 500"],
            "notes": "",
        },
    ),
    _spec(
        "abductive_reasoner",
        "AGI_Evolutive/reasoning/abduction.py",
        "Propose des hypothèses causales structurées.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Indique probabilité, mécanisme et tests recommandés.",
            "La probabilité doit être un nombre entre 0 et 1 (ex: 0.42).",
        ),
        example_output={
            "hypotheses": [
                {
                    "name": "surcharge proxy",
                    "probability": 0.6,
                    "mechanism": "trafic > capacité",
                    "tests": ["vérifier métriques proxy"],
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "rag_adaptive_controller",
        "AGI_Evolutive/retrieval/adaptive_controller.py",
        "Détermine les paramètres RAG optimaux selon la requête.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne poids_sparse, poids_dense et niveau_detail.",),
        example_output={
            "weights": {"sparse": 0.4, "dense": 0.6},
            "detail_level": "technique",
            "justification": "besoin d'analyse fine",
            "notes": "",
        },
    ),
    _spec(
        "ranker_model",
        "AGI_Evolutive/language/ranker.py",
        "Score les réponses candidates selon clarté et adéquation.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne un classement décroissant.",),
        example_output={
            "ranking": [
                {"id": "resp_a", "score": 0.78, "explanation": "répond directement"},
                {"id": "resp_b", "score": 0.55, "explanation": "trop vague"},
            ],
            "notes": "",
        },
    ),
    _spec(
        "style_policy",
        "AGI_Evolutive/language/style_policy.py",
        "Traduits des instructions stylistiques en curseurs numériques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne les curseurs chaleur/directivité/questionnement (0-1).",
        ),
        example_output={
            "directives": {"chaleur": 0.7, "directivite": 0.4, "questionnement": 0.6},
            "notes": "",
        },
    ),
    _spec(
        "cli_feedback",
        "AGI_Evolutive/main.py",
        "Interprète un feedback CLI et déduit l'émotion/urgence.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne 'urgency' parmi bas/moyen/haut.",),
        example_output={
            "sentiment": "mécontent",
            "urgency": "haut",
            "summary": "Utilisateur signale réponse trop lente",
            "notes": "",
        },
    ),
    _spec(
        "reasoning_strategies",
        "AGI_Evolutive/reasoning/strategies.py",
        "Choisit la stratégie de raisonnement et planifie les étapes.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris le plan et la confiance associée.",),
        example_output={
            "strategy": "analyse_causale",
            "steps": [
                {"description": "collecter métriques", "confidence": 0.8},
                {"description": "tester hypothèse proxy", "confidence": 0.7},
            ],
            "notes": "",
        },
    ),
    _spec(
        "models_intent",
        "AGI_Evolutive/models/intent.py",
        "Résume l'intention utilisateur avec horizon et justification.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Ajoute horizon (immédiat/court_terme/long_terme).",
        ),
        example_output={
            "intent": "réparer incident",
            "horizon": "immédiat",
            "justification": "impact direct sur utilisateurs",
            "notes": "",
        },
    ),
    _spec(
        "jsonl_logger",
        "AGI_Evolutive/runtime/logger.py",
        "Enrichit un événement avant écriture JSONL.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Fournis tags et résumé court.",),
        example_output={
            "event": "incident_api",
            "summary": "Erreur 500 détectée",
            "tags": ["incident", "api"],
            "notes": "",
        },
    ),
    _spec(
        "light_scheduler",
        "AGI_Evolutive/light_scheduler.py",
        "Explique les ajustements d'intervalle pour un job léger.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne intervalle_suggere en secondes.",),
        example_output={
            "job_id": "heartbeat",
            "interval_suggere": 45,
            "reason": "charge CPU modérée",
            "notes": "",
        },
    ),
    _spec(
        "scheduler",
        "AGI_Evolutive/runtime/scheduler.py",
        "Analyse un job de fond et ajuste sa politique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Retourne recommandation sur la politique (conserver/ralentir/accélérer).",
        ),
        example_output={
            "job": "refresh-metrics",
            "policy": "accélérer",
            "justification": "derives détectées",
            "notes": "",
        },
    ),
    _spec(
        "reward_engine",
        "AGI_Evolutive/cognition/reward_engine.py",
        "Évalue le feedback utilisateur et calcule la valence.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Identifie l'ironie éventuelle.",),
        example_output={
            "valence": -0.6,
            "irony_detected": True,
            "justification": "ton sarcastique concernant la lenteur",
            "notes": "",
        },
    ),
    _spec(
        "evolution_manager",
        "AGI_Evolutive/cognition/evolution_manager.py",
        "Analyse les tendances longue durée et suggère des actions.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue horizon et métriques impactées.",),
        example_output={
            "trend": "dérive latence",
            "horizon": "semaine",
            "affected_metrics": ["latence p95"],
            "suggested_actions": ["planifier optimisation cache"],
            "notes": "",
        },
    ),
    _spec(
        "thinking_monitor",
        "AGI_Evolutive/cognition/thinking_monitor.py",
        "Qualifie la qualité du raisonnement courant.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Donne un score 0-1 et les signaux clés.",),
        example_output={
            "score": 0.64,
            "signals": ["boucle logique détectée"],
            "notes": "",
        },
    ),
    _spec(
        "orchestrator_needs",
        "AGI_Evolutive/orchestrator.py",
        "Explique le protocole de besoins déclenché.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Précise message, durée et facteur d'intensité.",),
        example_output={
            "protocol": "pause_active",
            "duration": "5m",
            "intensity": 0.7,
            "message": "Prendre une pause courte pour réduire le stress",
            "notes": "",
        },
    ),
    _spec(
        "creativity_strategy_selector",
        "AGI_Evolutive/creativity/__init__.py",
        "Choisit et décrit la stratégie créative appropriée.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Justifie la stratégie et propose un variant.",),
        example_output={
            "strategy": "combinaison_concepts",
            "justification": "besoin d'idées hybrides",
            "variant": "associer incidents passés et tutoriels",
            "notes": "",
        },
    ),
    _spec(
        "context_feature_encoder",
        "AGI_Evolutive/learning/__init__.py",
        "Encode une expérience en features sémantiques.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne vecteur dense + tags.",),
        example_output={
            "features": [0.21, 0.05, -0.14],
            "tags": ["incident", "latence"],
            "notes": "",
        },
    ),
    _spec(
        "perception_module",
        "AGI_Evolutive/perception/__init__.py",
        "Résume les observations multi-modales et ajuste paramètres.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Propose ajustements pour sensibilité et fenêtre.",),
        example_output={
            "observations": [
                {"type": "audio", "issue": "bruit élevé"}
            ],
            "recommended_settings": {"sensibility": 0.6, "window_seconds": 8},
            "notes": "",
        },
    ),
    _spec(
        "language_nlg",
        "AGI_Evolutive/language/nlg.py",
        "Génère la réponse finale polie en respectant le contrat social.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue sections introduction, corps, conclusion.",),
        example_output={
            "introduction": "Merci pour les détails sur l'incident.",
            "body": "Voici le plan d'action proposé...",
            "conclusion": "Je reste disponible pour suivre la résolution.",
            "notes": "",
        },
    ),
    _spec(
        "language_lexicon",
        "AGI_Evolutive/language/lexicon.py",
        "Propose des variantes lexicales pertinentes et des collocations utiles.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=(
            "Retourne 'synonyms' (liste), 'collocations' (liste optionnelle) et le registre conseillé.",
        ),
        example_output={
            "synonyms": ["marge de progression", "axe d'amélioration"],
            "collocations": ["plan d'amélioration continue"],
            "register": "professionnel",
            "notes": "",
        },
    ),
    _spec(
        "language_renderer",
        "AGI_Evolutive/language/renderer.py",
        "Affiner la réponse finale en respectant le style et le contexte.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=(
            "Retourne un champ 'revision' (texte final) et 'lexicon_updates' si pertinent.",
        ),
        example_output={
            "revision": "Voici une proposition reformulée avec empathie.",
            "lexicon_updates": ["vision holistique"],
            "notes": "Accentuer la gratitude en ouverture.",
        },
    ),
    _spec(
        "language_voice",
        "AGI_Evolutive/language/voice.py",
        "Ajuste les curseurs de voix selon le feedback récent.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne variations proposées sur les knobs.",),
        example_output={
            "adjustments": {
                "warmth": +0.1,
                "conciseness": -0.05,
                "energy": +0.2,
            },
            "notes": "",
        },
    ),
    _spec(
        "action_interface",
        "AGI_Evolutive/io/action_interface.py",
        "Évalue les actions possibles et fournit un score sémantique.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Indique impact, effort, risque pour chaque action.",),
        example_output={
            "actions": [
                {
                    "name": "analyser_logs",
                    "impact": 0.8,
                    "effort": 0.3,
                    "risk": 0.2,
                    "rationale": "identifie la cause",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "metacog_calibration",
        "AGI_Evolutive/metacog/calibration.py",
        "Évalue la calibration de confiance d'une réponse.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne confiance perçue et conseils d'ajustement.",),
        example_output={
            "perceived_confidence": 0.58,
            "calibration_bias": "sous-confiance",
            "adjustment_advice": "exprimer assurance modérée",
            "notes": "",
        },
    ),
    _spec(
        "dialogue_state",
        "AGI_Evolutive/language/dialogue_state.py",
        "Maintient l'état de dialogue avec slots et engagements.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Liste les engagements ouverts avec échéance.",),
        example_output={
            "state_summary": "Incident API en cours de traitement",
            "open_commitments": [
                {"commitment": "fournir update", "deadline": "2024-05-10T09:00:00"}
            ],
            "pending_questions": ["confirmation déploiement"],
            "notes": "",
        },
    ),
    _spec(
        "cognition_proposer",
        "AGI_Evolutive/cognition/proposer.py",
        "Suggère des ajustements du self-model basés sur erreurs récentes.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Associe chaque suggestion à une cause identifiée.",),
        example_output={
            "suggestions": [
                {
                    "target": "persona.analytics",
                    "adjustment": "accentuer curiosité",
                    "cause": "erreurs d'anticipation incident",
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "homeostasis",
        "AGI_Evolutive/cognition/homeostasis.py",
        "Analyse les feedbacks pour ajuster les drives et rewards.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne deltas pour chaque drive (0-1).",),
        example_output={
            "drive_updates": {
                "competence": +0.1,
                "autonomie": -0.05,
            },
            "reward_signal": 0.3,
            "notes": "",
        },
    ),
    _spec(
        "trigger_router",
        "AGI_Evolutive/cognition/trigger_router.py",
        "Choisit les pipelines à activer pour un trigger.",
        AVAILABLE_MODELS["fast"],
        extra_instructions=("Retourne pipelines principaux et secondaires.",),
        example_output={
            "trigger": "THREAT",
            "pipelines": ["defense", "analyse_incident"],
            "secondary": ["demande_contexte"],
            "notes": "",
        },
    ),
    _spec(
        "episodic_linker",
        "AGI_Evolutive/memory/episodic_linker.py",
        "Identifie les liens causaux entre souvenirs.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Décris type_lien et confiance.",),
        example_output={
            "links": [
                {
                    "from": "incident_detecte",
                    "to": "redemarrage_proxy",
                    "type_lien": "cause",
                    "confidence": 0.7,
                }
            ],
            "notes": "",
        },
    ),
    _spec(
        "identity_mission",
        "AGI_Evolutive/cognition/identity_mission.py",
        "Met à jour la mission d'identité avec recommandations.",
        AVAILABLE_MODELS["reasoning"],
        extra_instructions=("Inclue axes prioritaire/support/vision.",),
        example_output={
            "mission": {
                "prioritaire": "maintenir disponibilité",
                "support": "apprentissage incident",
                "vision": "fiabilité durable",
            },
            "notes": "",
        },
    ),
)


SPEC_BY_KEY = {spec.key: spec for spec in LLM_INTEGRATION_SPECS}


def get_spec(key: str) -> LLMIntegrationSpec:
    try:
        return SPEC_BY_KEY[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown LLM integration spec: {key}") from exc


__all__ = [
    "AVAILABLE_MODELS",
    "LLM_INTEGRATION_SPECS",
    "LLMIntegrationSpec",
    "SPEC_BY_KEY",
    "get_spec",
]

