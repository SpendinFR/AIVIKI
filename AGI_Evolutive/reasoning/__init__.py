# reasoning/__init__.py
"""
Système de Raisonnement Complet de l'AGI Évolutive
Raisonnement causal, analogique, abductif, contrefactuel et probabiliste intégrés
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math
import random
from collections import defaultdict, deque
import networkx as nx

class ReasoningType(Enum):
    """Types de raisonnement"""
    DEDUCTIVE = "déductif"
    INDUCTIVE = "inductif"
    ABDUCTIVE = "abductif"
    ANALOGICAL = "analogique"
    CAUSAL = "causal"
    COUNTERFACTUAL = "contrefactuel"
    PROBABILISTIC = "probabiliste"
    TEMPORAL = "temporel"

class InferenceStrength(Enum):
    """Force des inférences"""
    WEAK = "faible"
    MODERATE = "modéré"
    STRONG = "fort"
    CERTAIN = "certain"

@dataclass
class ReasoningStep:
    """Étape individuelle dans un processus de raisonnement"""
    step_type: str
    premises: List[Any]
    conclusion: Any
    confidence: float
    reasoning_type: ReasoningType
    timestamp: float
    justification: str

@dataclass
class CausalModel:
    """Modèle causal d'un domaine"""
    causes: Dict[str, List[str]]
    effects: Dict[str, List[str]]
    strengths: Dict[Tuple[str, str], float]
    mechanisms: Dict[Tuple[str, str], str]
    temporal_constraints: Dict[Tuple[str, str], str]

@dataclass
class AnalogicalMapping:
    """Mapping analogique entre deux domaines"""
    source_domain: str
    target_domain: str
    correspondences: Dict[str, str]
    strength: float
    validity: float

class ReasoningSystem:
    """
    Système de raisonnement multi-modal inspiré des processus cognitifs humains
    Intègre différents types de raisonnement avec métacognition
    """
    
    def __init__(self, cognitive_architecture=None, memory_system=None, perception_system=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system
        self.perception_system = perception_system
        self.creation_time = time.time()

        # ——— LIAISONS INTER-MODULES ———
        if self.cognitive_architecture is not None:
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.metacognition = getattr(self.cognitive_architecture, "metacognition", None)
            self.world_model = getattr(self.cognitive_architecture, "world_model", None)

        
        # === MOTEURS DE RAISONNEMENT SPÉCIALISÉS ===
        self.reasoning_engines = {
            ReasoningType.DEDUCTIVE: DeductiveReasoner(),
            ReasoningType.INDUCTIVE: InductiveReasoner(),
            ReasoningType.ABDUCTIVE: AbductiveReasoner(),
            ReasoningType.ANALOGICAL: AnalogicalReasoner(),
            ReasoningType.CAUSAL: CausalReasoner(),
            ReasoningType.COUNTERFACTUAL: CounterfactualReasoner(),
            ReasoningType.PROBABILISTIC: ProbabilisticReasoner(),
            ReasoningType.TEMPORAL: TemporalReasoner()
        }
        
        # === BASE DE CONNAISSANCES INFÉRENTIELLE ===
        self.inferential_knowledge = {
            "logical_rules": {},
            "causal_models": {},
            "probabilistic_models": {},
            "analogical_mappings": {},
            "temporal_relations": {}
        }
        
        # === MÉTACOGNITION DU RAISONNEMENT ===
        self.metacognitive_monitoring = {
            "confidence_calibration": {},
            "error_detection": {},
            "strategy_selection": {},
            "resource_allocation": {}
        }
        
        # === HEURISTIQUES ET BIAS ===
        self.reasoning_heuristics = {
            "availability": AvailabilityHeuristic(),
            "representativeness": RepresentativenessHeuristic(),
            "anchoring": AnchoringHeuristic(),
            "confirmation_bias": ConfirmationBias()
        }
        
        # === PROCESSUS DE CONTRÔLE ===
        self.executive_control = {
            "goal_management": GoalManager(),
            "attention_allocation": AttentionAllocator(),
            "inhibition_control": InhibitionController(),
            "cognitive_flexibility": FlexibilityManager()
        }
        
        # === HISTORIQUE DE RAISONNEMENT ===
        self.reasoning_history = {
            "recent_inferences": deque(maxlen=100),
            "successful_strategies": {},
            "common_errors": {},
            "learning_trajectory": []
        }
        
        # === PARAMÈTRES DE RAISONNEMENT ===
        self.reasoning_parameters = {
            "default_confidence_threshold": 0.7,
            "max_reasoning_depth": 5,
            "working_memory_capacity": 4,
            "cognitive_load_threshold": 0.8,
            "uncertainty_tolerance": 0.3
        }
        
        # === COMPÉTENCES DE RAISONNEMENT INNÉES ===
        self._initialize_innate_reasoning()
        
        print("🧠 Système de raisonnement initialisé")
    
    def _initialize_innate_reasoning(self):
        """Initialise les compétences de raisonnement innées"""
        
        # Règles logiques fondamentales
        innate_rules = {
            "modus_ponens": {
                "premise_pattern": ["If A then B", "A"],
                "conclusion": "B",
                "certainty": 0.95
            },
            "transitivity": {
                "premise_pattern": ["A > B", "B > C"],
                "conclusion": "A > C", 
                "certainty": 0.9
            },
            "causality_basic": {
                "premise_pattern": ["A causes B", "A occurs"],
                "conclusion": "B likely occurs",
                "certainty": 0.8
            }
        }
        
        self.inferential_knowledge["logical_rules"] = innate_rules
        
        # Modèles causaux fondamentaux
        innate_causal_models = {
            "physical_intuitions": {
                "causes": {
                    "dropping": ["falling", "acceleration"],
                    "pushing": ["movement", "acceleration"]
                },
                "effects": {
                    "falling": ["impact", "damage"],
                    "movement": ["displacement", "momentum"]
                }
            }
        }
        
        self.inferential_knowledge["causal_models"] = innate_causal_models
        
        # Heuristiques innées
        self.reasoning_heuristics["availability"].base_level = 0.6
        self.reasoning_heuristics["representativeness"].base_level = 0.7
    
    def reason_about_problem(self, problem_statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Résout un problème en utilisant plusieurs types de raisonnement
        """
        reasoning_start = time.time()
        
        # === PHASE 1: ANALYSE DU PROBLÈME ===
        problem_analysis = self._analyze_problem(problem_statement, context)
        
        # === PHASE 2: SÉLECTION DES STRATÉGIES ===
        selected_strategies = self._select_reasoning_strategies(problem_analysis)
        
        # === PHASE 3: APPLICATION DES STRATÉGIES ===
        reasoning_results = {}
        for strategy in selected_strategies:
            engine = self.reasoning_engines.get(strategy)
            if engine:
                try:
                    result = engine.solve(problem_analysis, context)
                    reasoning_results[strategy] = result
                except Exception as e:
                    print(f"Erreur dans le raisonnement {strategy}: {e}")
        
        # === PHASE 4: INTÉGRATION DES RÉSULTATS ===
        integrated_solution = self._integrate_reasoning_results(reasoning_results, problem_analysis)
        
        # === PHASE 5: ÉVALUATION MÉTACOGNITIVE ===
        metacognitive_evaluation = self._evaluate_reasoning_process(
            integrated_solution, reasoning_results, problem_analysis
        )
        
        # === PHASE 6: APPRENTISSAGE ===
        self._learn_from_reasoning_episode(
            problem_analysis, selected_strategies, reasoning_results, integrated_solution
        )
        
        reasoning_time = time.time() - reasoning_start
        
        return {
            "problem": problem_statement,
            "solution": integrated_solution,
            "reasoning_strategies": [s.value for s in selected_strategies],
            "confidence": integrated_solution.get("confidence", 0.5),
            "reasoning_time": reasoning_time,
            "metacognitive_evaluation": metacognitive_evaluation,
            "reasoning_steps": self._extract_reasoning_steps(reasoning_results)
        }
    
    def _analyze_problem(self, problem_statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la structure et les exigences du problème"""
        analysis = {
            "problem_type": self._classify_problem_type(problem_statement),
            "complexity": self._assess_problem_complexity(problem_statement),
            "domain": self._identify_problem_domain(problem_statement),
            "known_information": self._extract_known_information(problem_statement, context),
            "unknown_target": self._identify_unknown_target(problem_statement),
            "constraints": self._identify_constraints(problem_statement, context)
        }
        
        # Recherche de problèmes similaires dans la mémoire
        if self.memory_system:
            similar_problems = self._find_similar_problems(problem_statement)
            analysis["similar_problems"] = similar_problems
        
        return analysis
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classe le type de problème"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["why", "cause", "reason"]):
            return "causal_explanation"
        elif any(word in problem_lower for word in ["how", "process", "method"]):
            return "procedural"
        elif any(word in problem_lower for word in ["what if", "would have", "counterfactual"]):
            return "counterfactual"
        elif any(word in problem_lower for word in ["like", "similar", "analogy"]):
            return "analogical"
        elif any(word in problem_lower for word in ["probability", "likely", "chance"]):
            return "probabilistic"
        else:
            return "general_inference"
    
    def _assess_problem_complexity(self, problem: str) -> float:
        """Évalue la complexité du problème"""
        complexity_factors = []
        
        # Longueur du problème
        word_count = len(problem.split())
        complexity_factors.append(min(word_count / 50, 1.0))
        
        # Nombre de concepts impliqués
        concept_indicators = ["because", "therefore", "however", "although", "despite"]
        concept_count = sum(1 for indicator in concept_indicators if indicator in problem.lower())
        complexity_factors.append(min(concept_count / 5, 1.0))
        
        return np.mean(complexity_factors) if complexity_factors else 0.5
    
    def _identify_problem_domain(self, problem: str) -> str:
        """Identifie le domaine du problème"""
        domain_keywords = {
            "physics": ["force", "motion", "energy", "gravity"],
            "social": ["person", "feel", "think", "want"],
            "mathematical": ["number", "calculate", "equation", "quantity"],
            "temporal": ["time", "before", "after", "while"]
        }
        
        problem_lower = problem.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in problem_lower for keyword in keywords):
                return domain
        
        return "general"
    
    def _extract_known_information(self, problem: str, context: Dict) -> List[str]:
        """Extrait les informations connues du problème"""
        known_info = []
        
        # Extraction basique des prémisses
        sentences = problem.split('.')
        for sentence in sentences:
            if not any(word in sentence.lower() for word in ["?", "find", "determine", "what", "why"]):
                known_info.append(sentence.strip())
        
        # Ajout du contexte
        if context:
            known_info.extend([f"{k}: {v}" for k, v in context.items()])
        
        return known_info
    
    def _identify_unknown_target(self, problem: str) -> str:
        """Identifie ce qui doit être trouvé/résolu"""
        question_words = ["what", "why", "how", "when", "where", "who"]
        problem_lower = problem.lower()
        
        for word in question_words:
            if word in problem_lower:
                # Extraire la partie après le mot question
                start_idx = problem_lower.find(word) + len(word)
                return problem[start_idx:].strip()
        
        return "unknown_target"
    
    def _identify_constraints(self, problem: str, context: Dict) -> List[str]:
        """Identifie les contraintes du problème"""
        constraints = []
        
        # Contraintes linguistiques
        constraint_indicators = ["must", "cannot", "only", "never", "always", "constraint"]
        for indicator in constraint_indicators:
            if indicator in problem.lower():
                constraints.append(f"linguistic: {indicator}")
        
        # Contraintes du contexte
        if "constraints" in context:
            constraints.extend(context["constraints"])
        
        return constraints
    
    def _find_similar_problems(self, problem: str) -> List[Dict[str, Any]]:
        """Trouve des problèmes similaires dans la mémoire"""
        # Intégration avec le système de mémoire
        if self.memory_system:
            try:
                retrieval_result = self.memory_system.retrieve_memories(
                    cues={"content": problem, "type": "problem"},
                    max_results=3
                )
                return [{"content": mem.content, "similarity": 0.7} for mem in retrieval_result.memory_traces]
            except:
                pass
        
        return []
    
    def _select_reasoning_strategies(self, problem_analysis: Dict[str, Any]) -> List[ReasoningType]:
        """Sélectionne les stratégies de raisonnement appropriées"""
        strategies = []
        problem_type = problem_analysis["problem_type"]
        complexity = problem_analysis["complexity"]
        
        # Stratégies basées sur le type de problème
        strategy_mapping = {
            "causal_explanation": [ReasoningType.CAUSAL, ReasoningType.ABDUCTIVE],
            "procedural": [ReasoningType.ANALOGICAL, ReasoningType.INDUCTIVE],
            "counterfactual": [ReasoningType.COUNTERFACTUAL, ReasoningType.TEMPORAL],
            "analogical": [ReasoningType.ANALOGICAL, ReasoningType.INDUCTIVE],
            "probabilistic": [ReasoningType.PROBABILISTIC, ReasoningType.INDUCTIVE],
            "general_inference": [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE, ReasoningType.ABDUCTIVE]
        }
        
        strategies.extend(strategy_mapping.get(problem_type, [ReasoningType.DEDUCTIVE]))
        
        # Ajout de stratégies basées sur la complexité
        if complexity > 0.7:
            strategies.append(ReasoningType.ANALOGICAL)  # Pour les problèmes complexes
        
        if problem_analysis.get("similar_problems"):
            strategies.append(ReasoningType.ANALOGICAL)
        
        # Élimination des doublons
        return list(set(strategies))
    
    def _integrate_reasoning_results(self, results: Dict[ReasoningType, Any], 
                                   problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Intègre les résultats de différents types de raisonnement"""
        if not results:
            return {"solution": "Aucune solution trouvée", "confidence": 0.0}
        
        # Collecte de toutes les solutions proposées
        all_solutions = []
        for reasoning_type, result in results.items():
            if "solution" in result:
                all_solutions.append({
                    "solution": result["solution"],
                    "confidence": result.get("confidence", 0.5),
                    "reasoning_type": reasoning_type,
                    "justification": result.get("justification", "")
                })
        
        if not all_solutions:
            return {"solution": "Aucune solution viable", "confidence": 0.0}
        
        # Sélection de la solution avec la plus haute confiance
        best_solution = max(all_solutions, key=lambda x: x["confidence"])
        
        # Vérification de la cohérence entre les solutions
        consistency_score = self._calculate_solution_consistency(all_solutions)
        
        # Ajustement de la confiance basé sur la cohérence
        adjusted_confidence = best_solution["confidence"] * consistency_score
        
        return {
            "solution": best_solution["solution"],
            "confidence": adjusted_confidence,
            "primary_reasoning_type": best_solution["reasoning_type"],
            "alternative_solutions": all_solutions,
            "consistency_score": consistency_score,
            "integrated_justification": self._generate_integrated_justification(all_solutions)
        }
    
    def _calculate_solution_consistency(self, solutions: List[Dict]) -> float:
        """Calcule la cohérence entre différentes solutions"""
        if len(solutions) <= 1:
            return 1.0
        
        # Conversion des solutions en vecteurs pour comparaison
        solution_vectors = []
        for sol in solutions:
            # Méthode basique de vectorisation
            solution_text = str(sol["solution"]).lower()
            vector = [1 if word in solution_text else 0 for word in ["yes", "no", "true", "false", "possible", "impossible"]]
            solution_vectors.append(vector)
        
        # Calcul des similarités par paires
        similarities = []
        for i in range(len(solution_vectors)):
            for j in range(i + 1, len(solution_vectors)):
                sim = self._cosine_similarity(solution_vectors[i], solution_vectors[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _generate_integrated_justification(self, solutions: List[Dict]) -> str:
        """Génère une justification intégrée pour la solution finale"""
        justifications = []
        
        for sol in solutions:
            reasoning_type = sol["reasoning_type"].value
            justification = sol.get("justification", "Aucune justification fournie")
            confidence = sol.get("confidence", 0.5)
            
            justifications.append(
                f"Raisonnement {reasoning_type} (confiance: {confidence:.2f}): {justification}"
            )
        
        return " | ".join(justifications)
    
    def _evaluate_reasoning_process(self, solution: Dict[str, Any], 
                                  results: Dict[ReasoningType, Any],
                                  problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue métacognitivement le processus de raisonnement"""
        evaluation = {
            "overall_confidence": solution.get("confidence", 0.0),
            "strategy_effectiveness": {},
            "cognitive_load": self._assess_cognitive_load(results),
            "error_likelihood": self._estimate_error_likelihood(results),
            "learning_opportunities": []
        }
        
        # Évaluation de l'efficacité des stratégies
        for reasoning_type, result in results.items():
            effectiveness = result.get("confidence", 0.0) * result.get("efficiency", 0.5)
            evaluation["strategy_effectiveness"][reasoning_type.value] = effectiveness
        
        # Identification des opportunités d’apprentissage
        if solution["confidence"] < self.reasoning_parameters["default_confidence_threshold"]:
            evaluation["learning_opportunities"].append("Améliorer la confiance dans les inférences")
        
        if evaluation["cognitive_load"] > self.reasoning_parameters["cognitive_load_threshold"]:
            evaluation["learning_opportunities"].append("Optimiser l'allocation des ressources cognitives")
        
        return evaluation
    
    def _assess_cognitive_load(self, results: Dict[ReasoningType, Any]) -> float:
        """Évalue la charge cognitive du processus de raisonnement"""
        load_factors = []
        
        for reasoning_type, result in results.items():
            # Charge basée sur la complexité du raisonnement
            complexity_weights = {
                ReasoningType.DEDUCTIVE: 0.7,
                ReasoningType.INDUCTIVE: 0.8,
                ReasoningType.ABDUCTIVE: 0.9,
                ReasoningType.ANALOGICAL: 0.8,
                ReasoningType.CAUSAL: 0.7,
                ReasoningType.COUNTERFACTUAL: 0.9,
                ReasoningType.PROBABILISTIC: 0.8,
                ReasoningType.TEMPORAL: 0.7
            }
            
            base_load = complexity_weights.get(reasoning_type, 0.5)
            processing_time = result.get("processing_time", 0.0)
            
            # Ajustement basé sur le temps de traitement
            time_adjustment = min(processing_time / 10.0, 1.0)  # Normalisation
            adjusted_load = base_load * (1.0 + time_adjustment)
            
            load_factors.append(adjusted_load)
        
        return np.mean(load_factors) if load_factors else 0.5
    
    def _estimate_error_likelihood(self, results: Dict[ReasoningType, Any]) -> float:
        """Estime la probabilité d'erreur dans le raisonnement"""
        error_indicators = []
        
        for reasoning_type, result in results.items():
            confidence = result.get("confidence", 0.5)
            
            # Faible confiance indique un risque d'erreur plus élevé
            error_risk = 1.0 - confidence
            error_indicators.append(error_risk)
            
            # Vérification de la cohérence interne
            if "internal_consistency" in result:
                consistency = result["internal_consistency"]
                error_indicators.append(1.0 - consistency)
        
        return np.mean(error_indicators) if error_indicators else 0.3
    
    def _learn_from_reasoning_episode(self, problem_analysis: Dict[str, Any],
                                    strategies: List[ReasoningType],
                                    results: Dict[ReasoningType, Any],
                                    final_solution: Dict[str, Any]):
        """Apprend de l'épisode de raisonnement pour améliorer les futures performances"""
        
        # Enregistrement dans l'historique
        episode_record = {
            "timestamp": time.time(),
            "problem_type": problem_analysis["problem_type"],
            "strategies_used": [s.value for s in strategies],
            "successful_strategies": [],
            "final_confidence": final_solution.get("confidence", 0.0),
            "learning_insights": []
        }
        
        # Identification des stratégies efficaces
        for reasoning_type, result in results.items():
            if result.get("confidence", 0.0) > self.reasoning_parameters["default_confidence_threshold"]:
                episode_record["successful_strategies"].append(reasoning_type.value)
        
        # Génération d'insights d’apprentissage
        if final_solution["confidence"] > 0.8:
            episode_record["learning_insights"].append(
                f"Stratégies efficaces pour les problèmes de type {problem_analysis['problem_type']}"
            )
        else:
            episode_record["learning_insights"].append(
                f"Besoin de meilleures stratégies pour les problèmes de type {problem_analysis['problem_type']}"
            )
        
        # Mise à jour de l'historique
        self.reasoning_history["recent_inferences"].append(episode_record)
        self.reasoning_history["learning_trajectory"].append({
            "timestamp": time.time(),
            "confidence": final_solution.get("confidence", 0.0),
            "problem_complexity": problem_analysis.get("complexity", 0.5)
        })
        
        # Mise à jour des heuristiques
        self._update_reasoning_heuristics(episode_record)
    
    def _update_reasoning_heuristics(self, episode_record: Dict[str, Any]):
        """Met à jour les heuristiques basé sur l'expérience"""
        # Ajustement de la disponibilité basé sur la récence
        for heuristic_name, heuristic in self.reasoning_heuristics.items():
            if hasattr(heuristic, 'update_based_on_experience'):
                heuristic.update_based_on_experience(episode_record)
    
    def _extract_reasoning_steps(self, results: Dict[ReasoningType, Any]) -> List[ReasoningStep]:
        """Extrait les étapes de raisonnement détaillées"""
        steps = []
        
        for reasoning_type, result in results.items():
            if "reasoning_steps" in result:
                steps.extend(result["reasoning_steps"])
            else:
                # Création d'une étape générique si non fournie
                step = ReasoningStep(
                    step_type=reasoning_type.value,
                    premises=result.get("premises", []),
                    conclusion=result.get("solution", "Unknown"),
                    confidence=result.get("confidence", 0.5),
                    reasoning_type=reasoning_type,
                    timestamp=time.time(),
                    justification=result.get("justification", "No detailed steps available")
                )
                steps.append(step)
        
        return steps
    
    def perform_causal_analysis(self, event: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue une analyse causale approfondie"""
        causal_engine = self.reasoning_engines[ReasoningType.CAUSAL]
        return causal_engine.analyze_causes(event, context)
    
    def generate_analogies(self, target_domain: str, source_domains: List[str]) -> List[AnalogicalMapping]:
        """Génère des analogies entre domaines"""
        analogical_engine = self.reasoning_engines[ReasoningType.ANALOGICAL]
        return analogical_engine.find_analogies(target_domain, source_domains)
    
    def evaluate_counterfactuals(self, scenario: str, changes: List[str]) -> Dict[str, Any]:
        """Évalue des scénarios contrefactuels"""
        counterfactual_engine = self.reasoning_engines[ReasoningType.COUNTERFACTUAL]
        return counterfactual_engine.evaluate_scenario(scenario, changes)
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de raisonnement"""
        return {
            "total_reasoning_episodes": len(self.reasoning_history["recent_inferences"]),
            "average_confidence": self._calculate_average_confidence(),
            "strategy_preferences": self._analyze_strategy_preferences(),
            "learning_progress": self._assess_learning_progress(),
            "metacognitive_awareness": self._evaluate_metacognitive_awareness()
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calcule la confiance moyenne dans les raisonnements récents"""
        if not self.reasoning_history["recent_inferences"]:
            return 0.0
        
        confidences = [episode["final_confidence"] for episode in self.reasoning_history["recent_inferences"]]
        return np.mean(confidences)
    
    def _analyze_strategy_preferences(self) -> Dict[str, float]:
        """Analyze les préférences de stratégie"""
        strategy_counts = defaultdict(int)
        total_episodes = len(self.reasoning_history["recent_inferences"])
        
        if total_episodes == 0:
            return {}
        
        for episode in self.reasoning_history["recent_inferences"]:
            for strategy in episode["strategies_used"]:
                strategy_counts[strategy] += 1
        
        return {strategy: count / total_episodes for strategy, count in strategy_counts.items()}
    
    def _assess_learning_progress(self) -> float:
        """Évalue le progrès d’apprentissage"""
        trajectory = self.reasoning_history["learning_trajectory"]
        if len(trajectory) < 2:
            return 0.0
        
        # Tendance de la confiance au fil du temps
        recent_confidence = [point["confidence"] for point in trajectory[-5:]]
        older_confidence = [point["confidence"] for point in trajectory[:5]]
        
        if not older_confidence or not recent_confidence:
            return 0.0
        
        avg_recent = np.mean(recent_confidence)
        avg_older = np.mean(older_confidence)
        
        return avg_recent - avg_older  # Amélioration positive
    
    def _evaluate_metacognitive_awareness(self) -> float:
        """Évalue la conscience métacognitive"""
        # Basé sur la calibration confiance-précision
        if not self.reasoning_history["recent_inferences"]:
            return 0.5
        
        calibration_errors = []
        for episode in self.reasoning_history["recent_inferences"]:
            # Pour l'instant, estimation basique
            # Dans une implémentation réelle, on comparerait confiance et précision réelle
            expected_error = 1.0 - episode["final_confidence"]
            calibration_errors.append(expected_error)
        
        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.5
        metacognitive_awareness = 1.0 - avg_calibration_error
        
        return max(0.0, min(1.0, metacognitive_awareness))

# ===== MOTEURS DE RAISONNEMENT SPÉCIALISÉS =====

class DeductiveReasoner:
    """Raisonnement déductif - des prémisses générales à des conclusions spécifiques"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        premises = problem_analysis["known_information"]
        target = problem_analysis["unknown_target"]
        
        # Application de règles logiques
        conclusions = self._apply_logical_rules(premises)
        
        # Filtrage des conclusions pertinentes
        relevant_conclusions = [c for c in conclusions if self._is_relevant(c, target)]
        
        if relevant_conclusions:
            best_conclusion = max(relevant_conclusions, key=lambda x: x["confidence"])
            return {
                "solution": best_conclusion["statement"],
                "confidence": best_conclusion["confidence"],
                "reasoning_type": ReasoningType.DEDUCTIVE,
                "premises": premises,
                "justification": f"Déduit de {len(premises)} prémisses using {best_conclusion['rule_used']}"
            }
        else:
            return {
                "solution": "Aucune conclusion déductive trouvée",
                "confidence": 0.1,
                "reasoning_type": ReasoningType.DEDUCTIVE
            }
    
    def _apply_logical_rules(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Applique des règles logiques aux prémisses"""
        conclusions = []
        
        # Règle simple: si toutes les prémisses sont vraies, une conclusion générique est possible
        if premises:
            conclusions.append({
                "statement": "Une conclusion logique peut être tirée des prémisses données",
                "confidence": 0.7,
                "rule_used": "logical_inference"
            })
        
        return conclusions
    
    def _is_relevant(self, conclusion: Dict[str, Any], target: str) -> bool:
        """Vérifie si une conclusion est pertinente pour la cible"""
        conclusion_text = conclusion["statement"].lower()
        target_lower = target.lower()
        
        # Vérification de chevauchement sémantique basique
        conclusion_words = set(conclusion_text.split())
        target_words = set(target_lower.split())
        
        return len(conclusion_words & target_words) > 0

class InductiveReasoner:
    """Raisonnement inductif - de spécifiques à généraux"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        specific_observations = problem_analysis["known_information"]
        
        # Généralisation à partir d'observations
        generalizations = self._generalize_from_observations(specific_observations)
        
        if generalizations:
            best_generalization = max(generalizations, key=lambda x: x["confidence"])
            return {
                "solution": best_generalization["pattern"],
                "confidence": best_generalization["confidence"],
                "reasoning_type": ReasoningType.INDUCTIVE,
                "observations_count": len(specific_observations),
                "justification": f"Généralisé à partir de {len(specific_observations)} observations"
            }
        else:
            return {
                "solution": "Aucun pattern inductif détecté",
                "confidence": 0.2,
                "reasoning_type": ReasoningType.INDUCTIVE
            }
    
    def _generalize_from_observations(self, observations: List[str]) -> List[Dict[str, Any]]:
        """Généralise des patterns à partir d'observations spécifiques"""
        generalizations = []
        
        if len(observations) >= 2:
            # Détection de patterns basique
            common_words = self._find_common_elements(observations)
            if common_words:
                generalizations.append({
                    "pattern": f"Pattern commun: {', '.join(common_words)}",
                    "confidence": min(0.3 + (len(observations) * 0.1), 0.8),
                    "support": len(observations)
                })
        
        return generalizations
    
    def _find_common_elements(self, observations: List[str]) -> List[str]:
        """Trouve des éléments communs dans les observations"""
        if not observations:
            return []
        
        # Extraction des mots communs
        all_words = [set(obs.lower().split()) for obs in observations]
        common_words = set.intersection(*all_words)
        
        # Filtrage des mots non significatifs
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        meaningful_words = common_words - stop_words
        
        return list(meaningful_words)

class AbductiveReasoner:
    """Raisonnement abductif - recherche de la meilleure explication"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        observations = problem_analysis["known_information"]
        target = problem_analysis["unknown_target"]
        
        # Génération d'hypothèses explicatives
        hypotheses = self._generate_explanatory_hypotheses(observations, target)
        
        if hypotheses:
            best_hypothesis = max(hypotheses, key=lambda x: x["explanatory_power"])
            return {
                "solution": best_hypothesis["explanation"],
                "confidence": best_hypothesis["explanatory_power"],
                "reasoning_type": ReasoningType.ABDUCTIVE,
                "alternative_explanations": len(hypotheses) - 1,
                "justification": f"Meilleure explication parmi {len(hypotheses)} hypothèses"
            }
        else:
            return {
                "solution": "Aucune explication abductive trouvée",
                "confidence": 0.1,
                "reasoning_type": ReasoningType.ABDUCTIVE
            }
    
    def _generate_explanatory_hypotheses(self, observations: List[str], target: str) -> List[Dict[str, Any]]:
        """Génère des hypothèses explicatives"""
        hypotheses = []
        
        # Hypothèse basique: causalité
        if any("cause" in obs.lower() for obs in observations):
            hypotheses.append({
                "explanation": "Relation causale possible",
                "explanatory_power": 0.6,
                "simplicity": 0.8
            })
        
        # Hypothèse basique: coïncidence
        hypotheses.append({
            "explanation": "Coïncidence ou corrélation non causale",
            "explanatory_power": 0.3,
            "simplicity": 0.9
        })
        
        return hypotheses

class AnalogicalReasoner:
    """Raisonnement analogique - transfert de solutions entre domaines similaires"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        target_problem = problem_analysis
        source_domains = problem_analysis.get("similar_problems", [])
        
        analogies = self._find_analogies(target_problem, source_domains)
        
        if analogies:
            best_analogy = max(analogies, key=lambda x: x["mapping_strength"])
            return {
                "solution": f"Solution analogique: {best_analogy['transferred_solution']}",
                "confidence": best_analogy["mapping_strength"],
                "reasoning_type": ReasoningType.ANALOGICAL,
                "source_domain": best_analogy["source_domain"],
                "justification": f"Basé sur l'analogie avec: {best_analogy['source_domain']}"
            }
        else:
            return {
                "solution": "Aucune analogie pertinente trouvée",
                "confidence": 0.2,
                "reasoning_type": ReasoningType.ANALOGICAL
            }
    
    def find_analogies(self, target_domain: str, source_domains: List[str]) -> List[AnalogicalMapping]:
        """Trouve des analogies entre domaines"""
        mappings = []
        
        for source in source_domains:
            similarity = self._calculate_domain_similarity(source, target_domain)
            if similarity > 0.5:
                mapping = AnalogicalMapping(
                    source_domain=source,
                    target_domain=target_domain,
                    correspondences=self._find_correspondences(source, target_domain),
                    strength=similarity,
                    validity=0.7
                )
                mappings.append(mapping)
        
        return mappings
    
    def _find_analogies(self, target_problem: Dict[str, Any], source_domains: List[Dict]) -> List[Dict[str, Any]]:
        """Trouve des analogies pour un problème donné"""
        analogies = []
        
        for source in source_domains:
            similarity = self._calculate_problem_similarity(target_problem, source)
            if similarity > 0.4:
                analogies.append({
                    "source_domain": source.get("content", "unknown"),
                    "transferred_solution": "Solution adaptée du domaine source",
                    "mapping_strength": similarity,
                    "structural_alignment": 0.6
                })
        
        return analogies
    
    def _calculate_problem_similarity(self, problem1: Dict, problem2: Dict) -> float:
        """Calcule la similarité entre deux problèmes"""
        similarity_factors = []
        
        # Similarité de type
        if problem1.get("problem_type") == problem2.get("problem_type"):
            similarity_factors.append(0.8)
        
        # Similarité de domaine
        if problem1.get("domain") == problem2.get("domain"):
            similarity_factors.append(0.7)
        
        # Similarité de complexité
        comp1 = problem1.get("complexity", 0.5)
        comp2 = problem2.get("complexity", 0.5)
        complexity_similarity = 1.0 - abs(comp1 - comp2)
        similarity_factors.append(complexity_similarity * 0.5)
        
        return np.mean(similarity_factors) if similarity_factors else 0.0
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calcule la similarité entre deux domaines"""
        # Implémentation basique
        if domain1 == domain2:
            return 0.9
        elif domain1.split()[0] == domain2.split()[0]:
            return 0.6
        else:
            return 0.3
    
    def _find_correspondences(self, source: str, target: str) -> Dict[str, str]:
        """Trouve des correspondances entre domaines"""
        # Correspondances basiques
        return {
            "problem": "problem",
            "solution": "solution",
            "constraint": "constraint"
        }

class CausalReasoner:
    """Raisonnement causal - relations cause-effet"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        event = problem_analysis["unknown_target"]
        known_info = problem_analysis["known_information"]
        
        causes = self._identify_possible_causes(event, known_info)
        mechanisms = self._infer_causal_mechanisms(event, causes)
        
        if causes:
            best_cause = max(causes, key=lambda x: x["causal_strength"])
            return {
                "solution": f"Cause probable: {best_cause['cause']}",
                "confidence": best_cause["causal_strength"],
                "reasoning_type": ReasoningType.CAUSAL,
                "causal_mechanism": mechanisms.get(best_cause["cause"], "inconnu"),
                "justification": f"Relation causale inférée avec force {best_cause['causal_strength']:.2f}"
            }
        else:
            return {
                "solution": "Aucune cause identifiable",
                "confidence": 0.1,
                "reasoning_type": ReasoningType.CAUSAL
            }
    
    def analyze_causes(self, event: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les causes d'un événement"""
        causes = self._identify_possible_causes(event, list(context.values()))
        mechanisms = self._infer_causal_mechanisms(event, causes)
        
        return {
            "event": event,
            "possible_causes": causes,
            "causal_mechanisms": mechanisms,
            "most_likely_cause": max(causes, key=lambda x: x["causal_strength"]) if causes else None
        }
    
    def _identify_possible_causes(self, event: str, context: List[str]) -> List[Dict[str, Any]]:
        """Identifie les causes possibles d'un événement"""
        causes = []
        
        # Causes basées sur le contexte
        for item in context:
            if self._could_be_cause(item, event):
                causes.append({
                    "cause": item,
                    "causal_strength": 0.6,
                    "temporal_relation": "avant",
                    "necessity": 0.5,
                    "sufficiency": 0.4
                })
        
        # Cause générique
        causes.append({
            "cause": "Facteurs inconnus ou complexes",
            "causal_strength": 0.3,
            "temporal_relation": "inconnue",
            "necessity": 0.2,
            "sufficiency": 0.1
        })
        
        return causes
    
    def _could_be_cause(self, potential_cause: str, effect: str) -> bool:
        """Détermine si quelque chose pourrait être une cause"""
        cause_words = potential_cause.lower().split()
        effect_words = effect.lower().split()
        
        # Vérification de chevauchement sémantique
        return len(set(cause_words) & set(effect_words)) > 0
    
    def _infer_causal_mechanisms(self, event: str, causes: List[Dict]) -> Dict[str, str]:
        """Infère les mécanismes causaux possibles"""
        mechanisms = {}
        
        for cause in causes:
            cause_text = cause["cause"]
            if "force" in cause_text.lower() or "push" in cause_text.lower():
                mechanisms[cause_text] = "mécanisme physique"
            elif "think" in cause_text.lower() or "want" in cause_text.lower():
                mechanisms[cause_text] = "mécanisme intentionnel"
            else:
                mechanisms[cause_text] = "mécanisme inconnu"
        
        return mechanisms

class CounterfactualReasoner:
    """Raisonnement contrefactuel - scénarios 'et si...'"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        scenario = problem_analysis["known_information"]
        changes = problem_analysis.get("constraints", [])
        
        counterfactual_scenarios = self._generate_counterfactuals(scenario, changes)
        
        if counterfactual_scenarios:
            best_scenario = max(counterfactual_scenarios, key=lambda x: x["plausibility"])
            return {
                "solution": f"Scénario contrefactuel: {best_scenario['outcome']}",
                "confidence": best_scenario["plausibility"],
                "reasoning_type": ReasoningType.COUNTERFACTUAL,
                "alternative_scenarios": len(counterfactual_scenarios),
                "justification": f"Scénario avec plausibilité {best_scenario['plausibility']:.2f}"
            }
        else:
            return {
                "solution": "Aucun scénario contrefactuel plausible",
                "confidence": 0.1,
                "reasoning_type": ReasoningType.COUNTERFACTUAL
            }
    
    def evaluate_scenario(self, scenario: str, changes: List[str]) -> Dict[str, Any]:
        """Évalue un scénario contrefactuel"""
        counterfactuals = self._generate_counterfactuals([scenario], changes)
        
        return {
            "original_scenario": scenario,
            "proposed_changes": changes,
            "counterfactual_outcomes": counterfactuals,
            "most_plausible": max(counterfactuals, key=lambda x: x["plausibility"]) if counterfactuals else None
        }
    
    def _generate_counterfactuals(self, scenario: List[str], changes: List[str]) -> List[Dict[str, Any]]:
        """Génère des scénarios contrefactuels"""
        counterfactuals = []
        
        for change in changes:
            # Scénario basé sur le changement
            outcome = self._simulate_counterfactual_outcome(scenario, change)
            plausibility = self._assess_counterfactual_plausibility(scenario, change, outcome)
            
            counterfactuals.append({
                "change": change,
                "outcome": outcome,
                "plausibility": plausibility,
                "causal_dependence": 0.6
            })
        
        return counterfactuals
    
    def _simulate_counterfactual_outcome(self, scenario: List[str], change: str) -> str:
        """Simule le résultat d'un changement contrefactuel"""
        return f"Si {change}, alors le résultat serait différent"
    
    def _assess_counterfactual_plausibility(self, scenario: List[str], change: str, outcome: str) -> float:
        """Évalue la plausibilité d'un scénario contrefactuel"""
        # Facteurs de plausibilité
        factors = []
        
        # Cohérence avec les lois physiques (estimation)
        factors.append(0.7)
        
        # Minimalité du changement
        change_complexity = len(change.split())
        factors.append(1.0 - min(change_complexity / 10, 0.8))
        
        return np.mean(factors)

class ProbabilisticReasoner:
    """Raisonnement probabiliste - incertitude et probabilités"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        uncertain_information = problem_analysis["known_information"]
        target = problem_analysis["unknown_target"]
        
        probabilities = self._estimate_probabilities(uncertain_information, target)
        
        if probabilities:
            best_prediction = max(probabilities.items(), key=lambda x: x[1])
            return {
                "solution": f"Probabilité la plus élevée: {best_prediction[0]} ({best_prediction[1]:.2f})",
                "confidence": best_prediction[1],
                "reasoning_type": ReasoningType.PROBABILISTIC,
                "probability_distribution": probabilities,
                "justification": f"Distribution probabiliste calculée sur {len(uncertain_information)} éléments"
            }
        else:
            return {
                "solution": "Incertitude trop élevée pour une prédiction",
                "confidence": 0.1,
                "reasoning_type": ReasoningType.PROBABILISTIC
            }
    
    def _estimate_probabilities(self, information: List[str], target: str) -> Dict[str, float]:
        """Estime les probabilités pour différentes outcomes"""
        probabilities = {}
        
        # Estimation basique basée sur la présence de mots-clés
        positive_indicators = ["likely", "probable", "usually", "often"]
        negative_indicators = ["unlikely", "improbable", "rarely", "seldom"]
        
        positive_count = sum(1 for info in information if any(indicator in info.lower() for indicator in positive_indicators))
        negative_count = sum(1 for info in information if any(indicator in info.lower() for indicator in negative_indicators))
        total_indicators = positive_count + negative_count
        
        if total_indicators > 0:
            positive_prob = positive_count / total_indicators
            probabilities["Outcome positif"] = positive_prob
            probabilities["Outcome négatif"] = 1.0 - positive_prob
        else:
            # Probabilités par défaut en cas d'information insuffisante
            probabilities["Outcome positif"] = 0.5
            probabilities["Outcome négatif"] = 0.5
        
        return probabilities

class TemporalReasoner:
    """Raisonnement temporel - relations temporelles et séquences"""
    
    def solve(self, problem_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        temporal_information = problem_analysis["known_information"]
        target = problem_analysis["unknown_target"]
        
        temporal_relations = self._infer_temporal_relations(temporal_information)
        sequence = self._construct_temporal_sequence(temporal_information)
        
        return {
            "solution": f"Séquence temporelle: {' -> '.join(sequence)}",
            "confidence": 0.7,
            "reasoning_type": ReasoningType.TEMPORAL,
            "temporal_relations": temporal_relations,
            "justification": f"Séquence construite à partir de {len(temporal_information)} éléments temporels"
        }
    
    def _infer_temporal_relations(self, information: List[str]) -> Dict[str, str]:
        """Infère les relations temporelles entre événements"""
        relations = {}
        
        # Détection basique de relations temporelles
        for info in information:
            if "before" in info.lower():
                relations[info] = "avant"
            elif "after" in info.lower():
                relations[info] = "après"
            elif "while" in info.lower() or "during" in info.lower():
                relations[info] = "simultané"
            else:
                relations[info] = "relation temporelle inconnue"
        
        return relations
    
    def _construct_temporal_sequence(self, information: List[str]) -> List[str]:
        """Construit une séquence temporelle à partir des informations"""
        # Séquence basée sur l'ordre d'apparition (simplifié)
        return [f"Événement {i+1}" for i in range(min(3, len(information)))]

# ===== HEURISTIQUES DE RAISONNEMENT =====

class AvailabilityHeuristic:
    """Heuristique de disponibilité - estimation basée sur ce qui vient facilement à l'esprit"""
    
    def __init__(self):
        self.base_level = 0.5
        self.recent_experiences = deque(maxlen=10)
    
    def estimate_probability(self, event: str) -> float:
        """Estime la probabilité basée sur la disponibilité"""
        availability_score = self._calculate_availability(event)
        return min(1.0, self.base_level + availability_score)
    
    def _calculate_availability(self, event: str) -> float:
        """Calcule le score de disponibilité d'un événement"""
        # Basé sur la récence et la fréquence
        recency_score = self._calculate_recency_score(event)
        frequency_score = self._calculate_frequency_score(event)
        
        return (recency_score + frequency_score) / 2
    
    def _calculate_recency_score(self, event: str) -> float:
        """Calcule le score de récence"""
        if not self.recent_experiences:
            return 0.0
        
        # Vérifie si l'événement est dans les expériences récentes
        event_lower = event.lower()
        for exp in self.recent_experiences:
            if event_lower in exp.lower():
                return 0.8
        
        return 0.0
    
    def _calculate_frequency_score(self, event: str) -> float:
        """Calcule le score de fréquence"""
        # Implémentation basique
        return 0.3
    
    def update_based_on_experience(self, experience: Dict[str, Any]):
        """Met à jour basé sur l'expérience"""
        self.recent_experiences.append(experience.get("problem_type", ""))

class RepresentativenessHeuristic:
    """Heuristique de représentativité - jugement basé sur la similarité avec un prototype"""
    
    def __init__(self):
        self.base_level = 0.5
        self.prototypes = {}
    
    def assess_similarity(self, instance: str, category: str) -> float:
        """Évalue la similarité avec une catégorie"""
        prototype = self.prototypes.get(category)
        if not prototype:
            return 0.5
        
        return self._calculate_similarity(instance, prototype)
    
    def _calculate_similarity(self, instance: str, prototype: str) -> float:
        """Calcule la similarité entre une instance et un prototype"""
        # Similarité basée sur le chevauchement de mots
        instance_words = set(instance.lower().split())
        prototype_words = set(prototype.lower().split())
        
        if not instance_words or not prototype_words:
            return 0.0
        
        intersection = instance_words & prototype_words
        union = instance_words | prototype_words
        
        return len(intersection) / len(union)

class AnchoringHeuristic:
    """Heuristique d'ancrage - influence des premières informations"""
    
    def __init__(self):
        self.anchors = {}
        self.adjustment_rate = 0.2
    
    def adjust_estimate(self, initial_estimate: float, new_information: str) -> float:
        """Ajuste une estimation basée sur de nouvelles informations"""
        anchor_influence = self._get_anchor_influence(new_information)
        adjustment = (initial_estimate * anchor_influence) * self.adjustment_rate
        
        return initial_estimate + adjustment
    
    def _get_anchor_influence(self, information: str) -> float:
        """Calcule l'influence d'une ancre"""
        # Influence basée sur la force perçue de l'information
        strength_indicators = ["certain", "definite", "proven", "established"]
        weakness_indicators = ["maybe", "possibly", "uncertain", "speculative"]
        
        info_lower = information.lower()
        strength_score = sum(1 for indicator in strength_indicators if indicator in info_lower)
        weakness_score = sum(1 for indicator in weakness_indicators if indicator in info_lower)
        
        net_strength = strength_score - weakness_score
        return max(-1.0, min(1.0, net_strength * 0.1))

class ConfirmationBias:
    """Biais de confirmation - tendance à chercher des informations confirmantes"""
    
    def __init__(self):
        self.strength = 0.3
        self.confirming_evidence_weight = 1.2
        self.disconfirming_evidence_weight = 0.8
    
    def weight_evidence(self, evidence: List[Dict], hypothesis: str) -> List[Dict]:
        """Pondère les preuves basé sur le biais de confirmation"""
        weighted_evidence = []
        
        for item in evidence:
            if self._supports_hypothesis(item, hypothesis):
                # Renforce les preuves confirmantes
                weighted_item = item.copy()
                weighted_item["weight"] = item.get("weight", 1.0) * self.confirming_evidence_weight
                weighted_evidence.append(weighted_item)
            else:
                # Diminue les preuves infirmantes
                weighted_item = item.copy()
                weighted_item["weight"] = item.get("weight", 1.0) * self.disconfirming_evidence_weight
                weighted_evidence.append(weighted_item)
        
        return weighted_evidence
    
    def _supports_hypothesis(self, evidence: Dict, hypothesis: str) -> bool:
        """Détermine si une preuve supporte l'hypothèse"""
        evidence_text = str(evidence.get("content", "")).lower()
        hypothesis_lower = hypothesis.lower()
        
        # Vérification basique de chevauchement sémantique
        evidence_words = set(evidence_text.split())
        hypothesis_words = set(hypothesis_lower.split())
        
        return len(evidence_words & hypothesis_words) > 0

# ===== CONTRÔLE EXÉCUTIF =====

class GoalManager:
    """Gestion des buts de raisonnement"""
    
    def __init__(self):
        self.active_goals = []
        self.goal_priorities = {}
    
    def set_reasoning_goal(self, goal: str, priority: float = 0.5):
        """Définit un but de raisonnement"""
        self.active_goals.append(goal)
        self.goal_priorities[goal] = priority
    
    def get_current_priority(self) -> Optional[str]:
        """Retourne le but prioritaire actuel"""
        if not self.active_goals:
            return None
        
        return max(self.active_goals, key=lambda g: self.goal_priorities.get(g, 0.0))

class AttentionAllocator:
    """Allocation de l'attention pendant le raisonnement"""
    
    def __init__(self):
        self.attention_budget = 1.0
        self.current_focus = None
    
    def allocate_attention(self, reasoning_components: List[str]) -> Dict[str, float]:
        """Alloue l'attention aux composants de raisonnement"""
        allocation = {}
        total_importance = sum(self._estimate_importance(comp) for comp in reasoning_components)
        
        for component in reasoning_components:
            importance = self._estimate_importance(component)
            allocation[component] = (importance / total_importance) * self.attention_budget
        
        return allocation
    
    def _estimate_importance(self, component: str) -> float:
        """Estime l'importance d'un composant de raisonnement"""
        importance_weights = {
            "premise_analysis": 0.8,
            "hypothesis_generation": 0.9,
            "evidence_evaluation": 0.7,
            "conclusion_drawing": 1.0
        }
        
        return importance_weights.get(component, 0.5)

class InhibitionController:
    """Contrôle de l'inhibition pendant le raisonnement"""
    
    def __init__(self):
        self.inhibition_strength = 0.7
        self.inhibited_paths = set()
    
    def should_inhibit(self, reasoning_path: str, context: Dict[str, Any]) -> bool:
        """Détermine si un chemin de raisonnement devrait être inhibé"""
        # Inhibition des chemins récemment échoués
        if reasoning_path in self.inhibited_paths:
            return True
        
        # Inhibition basée sur la complexité
        if self._is_overly_complex(reasoning_path):
            return True
        
        return False
    
    def _is_overly_complex(self, reasoning_path: str) -> bool:
        """Vérifie si un chemin de raisonnement est trop complexe"""
        complexity_indicators = ["multiple", "complex", "complicated", "convoluted"]
        path_lower = reasoning_path.lower()
        
        return any(indicator in path_lower for indicator in complexity_indicators)

class FlexibilityManager:
    """Gestion de la flexibilité cognitive"""
    
    def __init__(self):
        self.flexibility_level = 0.6
        self.strategy_switching_threshold = 0.3
    
    def should_switch_strategy(self, current_strategy: str, 
                             performance: float, 
                             alternatives: List[str]) -> bool:
        """Détermine s'il faut changer de stratégie"""
        if performance < self.strategy_switching_threshold:
            return True
        
        if len(alternatives) > 0 and self.flexibility_level > 0.7:
            return True
        
        return False

# Test du système de raisonnement
if __name__ == "__main__":
    print("🧠 TEST DU SYSTÈME DE RAISONNEMENT")
    print("=" * 50)
    
    # Création du système
    reasoning_system = ReasoningSystem()
    
    # Problèmes de test
    test_problems = [
        {
            "statement": "Si un objet est lâché, il tombe. Cet objet est lâché. Que se passe-t-il?",
            "context": {"domain": "physique", "certainty": "high"}
        },
        {
            "statement": "Pourquoi la glace fond-elle quand il fait chaud?",
            "context": {"domain": "physique", "constraints": ["température"]}
        },
        {
            "statement": "Quelle est la probabilité qu'il pleuve demain si le ciel est nuageux?",
            "context": {"domain": "météo", "uncertainty": "high"}
        }
    ]
    
    for i, problem_data in enumerate(test_problems):
        print(f"\n🎯 Problème {i+1}: {problem_data['statement']}")
        
        result = reasoning_system.reason_about_problem(
            problem_data["statement"],
            problem_data["context"]
        )
        
        print(f"Solution: {result['solution']}")
        print(f"Confiance: {result['confidence']:.2f}")
        print(f"Stratégies utilisées: {result['reasoning_strategies']}")
        print(f"Temps de raisonnement: {result['reasoning_time']:.3f}s")
    
    # Tests des fonctionnalités spécialisées
    print("\n🔍 TESTS SPÉCIALISÉS:")
    
    # Analyse causale
    causal_result = reasoning_system.perform_causal_analysis(
        "l'objet est tombé",
        {"contexte": "il a été lâché"}
    )
    print(f"Analyse causale: {causal_result['most_likely_cause']}")
    
    # Statistiques
    print("\n📊 STATISTIQUES DU SYSTÈME:")
    stats = reasoning_system.get_reasoning_stats()
    for key, value in stats.items():
        print(f" - {key}: {value}")