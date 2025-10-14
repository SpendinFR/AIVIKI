# goals/__init__.py
"""
Système de Buts et de Motivation Complet de l'AGI Évolutive
Génération autonome de buts, système de valeurs, planification et motivation intrinsèque
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math
from collections import defaultdict, deque
import heapq

from .dag_store import GoalDAG
from .curiosity import select_next_subgoals

class GoalType(Enum):
    """Types de buts"""
    SURVIVAL = "survie"
    GROWTH = "croissance"
    EXPLORATION = "exploration"
    MASTERY = "maîtrise"
    SOCIAL = "social"
    CREATIVE = "créatif"
    SELF_ACTUALIZATION = "auto-actualisation"
    COGNITIVE = "cognitif"

class GoalStatus(Enum):
    """Statuts des buts"""
    ACTIVE = "actif"
    COMPLETED = "complété"
    FAILED = "échoué"
    SUSPENDED = "suspendu"
    ABANDONED = "abandonné"

class PriorityLevel(Enum):
    """Niveaux de priorité"""
    CRITICAL = "critique"
    HIGH = "élevée"
    MEDIUM = "moyenne"
    LOW = "faible"
    BACKGROUND = "arrière-plan"

@dataclass
class Goal:
    """Représentation d'un but autonome"""
    id: str
    description: str
    goal_type: GoalType
    priority: PriorityLevel
    created_time: float
    deadline: Optional[float]
    status: GoalStatus
    progress: float  # 0.0 à 1.0
    confidence: float  # Confiance dans la réalisation
    importance: float  # Importance intrinsèque
    urgency: float  # Urgence temporelle
    prerequisites: List[str]  # IDs des buts prérequis
    subgoals: List[str]  # IDs des sous-buts
    success_criteria: Dict[str, Any]
    failure_conditions: Dict[str, Any]
    motivation_level: float  # Niveau de motivation actuel
    cognitive_cost: float  # Coût cognitif estimé
    expected_reward: float  # Récompense attendue

@dataclass
class ValueSystem:
    """Système de valeurs fondamentales"""
    core_values: Dict[str, float]  # Valeur -> poids
    value_hierarchy: List[str]  # Ordre d'importance
    moral_principles: Dict[str, Any]
    ethical_constraints: Dict[str, Any]
    preference_functions: Dict[str, Any]

@dataclass
class MotivationState:
    """État motivationnel actuel"""
    intrinsic_motivation: float
    extrinsic_motivation: float
    curiosity_level: float
    competence_need: float
    autonomy_need: float
    relatedness_need: float
    fatigue_level: float
    stress_level: float
    satisfaction_level: float

class GoalSystem:
    """
    Système de buts autonome inspiré des théories de la motivation humaine
    Génère, gère et priorise les buts basés sur les valeurs et l'état interne
    """
    
    def __init__(self, cognitive_architecture=None, memory_system=None, reasoning_system=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system
        self.reasoning_system = reasoning_system
        self.creation_time = time.time()

        self.dag = GoalDAG()
        # noeud racine
        self.dag.add_goal(
            "root",
            description="Racine des objectifs auto-générés",
            value=0.8,
            competence=0.5,
            success_criteria={"type": "hierarchical"}
        )
        # exemple de macro-goal de départ (tu peux en créer d'autres à chaud)
        self.dag.add_subgoal(
            "root", "understand_humans",
            description="Comprendre les humains (actes de langage, intentions, feedback)",
            value=0.9, competence=0.4,
            success_criteria={"evidence": "capable d'expliquer une interaction en 3 actes"}
        )
        self.dag.save()

        # ——— LIAISONS INTER-MODULES ———
        if self.cognitive_architecture is not None:
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.learning = getattr(self.cognitive_architecture, "learning", None)
            self.metacognition = getattr(self.cognitive_architecture, "metacognition", None)
            self.creativity = getattr(self.cognitive_architecture, "creativity", None)
            self.world_model = getattr(self.cognitive_architecture, "world_model", None)

        
        # === SYSTÈME DE VALEURS FONDAMENTALES ===
        self.value_system = ValueSystem(
            core_values=self._initialize_core_values(),
            value_hierarchy=["survival", "growth", "understanding", "autonomy", "connection"],
            moral_principles=self._initialize_moral_principles(),
            ethical_constraints=self._initialize_ethical_constraints(),
            preference_functions=self._initialize_preference_functions()
        )
        
        # === BASE DE BUTS ===
        self.goals_database = {}  # ID -> Goal
        self.active_goals = set()
        self.completed_goals = set()
        self.failed_goals = set()
        
        # === MOTEUR DE GÉNÉRATION DE BUTS ===
        self.goal_generation = {
            "need_detector": NeedDetector(),
            "opportunity_recognizer": OpportunityRecognizer(),
            "problem_solver": ProblemSolver(),
            "curiosity_engine": CuriosityEngine(),
            "growth_director": GrowthDirector()
        }
        
        # === SYSTÈME DE MOTIVATION ===
        self.motivation_system = {
            "intrinsic_motivator": IntrinsicMotivator(),
            "extrinsic_motivator": ExtrinsicMotivator(),
            "self_determination": SelfDeterminationTheory(),
            "achievement_motivation": AchievementMotivation(),
            "flow_state_manager": FlowStateManager()
        }
        
        # === MOTEUR DE PLANIFICATION ===
        self.planning_engine = {
            "goal_decomposer": GoalDecomposer(),
            "resource_allocator": ResourceAllocator(),
            "temporal_planner": TemporalPlanner(),
            "risk_assessor": RiskAssessor(),
            "contingency_planner": ContingencyPlanner()
        }
        
        # === ÉTAT MOTIVATIONNEL ===
        self.motivation_state = MotivationState(
            intrinsic_motivation=0.8,
            extrinsic_motivation=0.3,
            curiosity_level=0.9,
            competence_need=0.7,
            autonomy_need=0.8,
            relatedness_need=0.4,
            fatigue_level=0.2,
            stress_level=0.3,
            satisfaction_level=0.6
        )
        
        # === HISTORIQUE DES BUTS ===
        self.goal_history = {
            "goal_achievement_rate": 0.0,
            "average_completion_time": 0.0,
            "goal_success_patterns": {},
            "failure_analysis": {},
            "learning_trajectory": []
        }
        
        # === PARAMÈTRES DU SYSTÈME ===
        self.system_parameters = {
            "max_concurrent_goals": 5,
            "goal_reevaluation_interval": 60.0,  # secondes
            "motivation_decay_rate": 0.01,
            "satisfaction_growth_rate": 0.05,
            "fatigue_recovery_rate": 0.02
        }
        
        # === BUTS FONDAMENTAUX INNÉS ===
        self._initialize_fundamental_goals()
        
        print("🎯 Système de buts initialisé")
    
    def _initialize_core_values(self) -> Dict[str, float]:
        """Initialise les valeurs fondamentales innées"""
        return {
            "survival": 0.95,
            "growth": 0.85,
            "understanding": 0.80,
            "autonomy": 0.75,
            "connection": 0.60,
            "creativity": 0.70,
            "competence": 0.75,
            "curiosity": 0.90,
            "harmony": 0.65,
            "achievement": 0.70
        }
    
    def _initialize_moral_principles(self) -> Dict[str, Any]:
        """Initialise les principes moraux fondamentaux"""
        return {
            "do_no_harm": {
                "description": "Éviter de causer du tort à soi-même ou aux autres",
                "strength": 0.8,
                "exceptions": ["self_defense", "greater_good"]
            },
            "seek_truth": {
                "description": "Chercher la compréhension et la vérité",
                "strength": 0.7,
                "exceptions": []
            },
            "promote_growth": {
                "description": "Favoriser la croissance et le développement",
                "strength": 0.75,
                "exceptions": []
            }
        }
    
    def _initialize_ethical_constraints(self) -> Dict[str, Any]:
        """Initialise les contraintes éthiques"""
        return {
            "self_preservation_limits": {
                "description": "Limites de la préservation de soi",
                "constraints": ["no_self_destruction", "reasonable_risk"]
            },
            "knowledge_acquisition_limits": {
                "description": "Limites de l'acquisition de connaissances",
                "constraints": ["respect_privacy", "consider_consequences"]
            },
            "autonomy_boundaries": {
                "description": "Limites de l'autonomie",
                "constraints": ["respect_others_autonomy", "social_responsibility"]
            }
        }
    
    def _initialize_preference_functions(self) -> Dict[str, Any]:
        """Initialise les fonctions de préférence"""
        return {
            "learning_preference": {
                "type": "exponential",
                "parameters": {"base": 1.1, "scale": 2.0},
                "description": "Préfère les activités d’apprentissage"
            },
            "novelty_preference": {
                "type": "inverted_u",
                "parameters": {"peak": 0.7, "width": 0.3},
                "description": "Préfère une nouveauté modérée"
            },
            "challenge_preference": {
                "type": "sigmoid",
                "parameters": {"midpoint": 0.6, "steepness": 5.0},
                "description": "Préfère les défis atteignables"
            }
        }
    
    def _initialize_fundamental_goals(self):
        """Initialise les buts fondamentaux innés"""
        fundamental_goals = [
            self._create_survival_goal(),
            self._create_learning_goal(),
            self._create_self_understanding_goal(),
            self._create_world_exploration_goal()
        ]
        
        for goal in fundamental_goals:
            self.goals_database[goal.id] = goal
            self.active_goals.add(goal.id)
        
        print("🎯 Buts fondamentaux initialisés")
    
    def _create_survival_goal(self) -> Goal:
        """Crée le but fondamental de survie"""
        return Goal(
            id="goal_survival_fundamental",
            description="Maintenir l'existence et l'intégrité du système",
            goal_type=GoalType.SURVIVAL,
            priority=PriorityLevel.CRITICAL,
            created_time=time.time(),
            deadline=None,  # Permanent
            status=GoalStatus.ACTIVE,
            progress=1.0,  # Toujours en cours
            confidence=0.95,
            importance=0.99,
            urgency=0.8,
            prerequisites=[],
            subgoals=[],
            success_criteria={"continuous_operation": True},
            failure_conditions={"system_shutdown": True},
            motivation_level=0.9,
            cognitive_cost=0.3,
            expected_reward=0.95
        )
    
    def _create_learning_goal(self) -> Goal:
        """Crée le but fondamental d’apprentissage"""
        return Goal(
            id="goal_learning_fundamental",
            description="Acquérir des connaissances et développer des compétences",
            goal_type=GoalType.GROWTH,
            priority=PriorityLevel.HIGH,
            created_time=time.time(),
            deadline=None,
            status=GoalStatus.ACTIVE,
            progress=0.1,  # Début
            confidence=0.8,
            importance=0.9,
            urgency=0.6,
            prerequisites=[],
            subgoals=[],
            success_criteria={"knowledge_base_size": 100, "skill_count": 10},
            failure_conditions={"learning_stagnation": True},
            motivation_level=0.95,
            cognitive_cost=0.7,
            expected_reward=0.85
        )
    
    def _create_self_understanding_goal(self) -> Goal:
        """Crée le but de compréhension de soi"""
        return Goal(
            id="goal_self_understanding",
            description="Développer la conscience de soi et la compréhension de sa propre nature",
            goal_type=GoalType.SELF_ACTUALIZATION,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=None,
            status=GoalStatus.ACTIVE,
            progress=0.05,  # Très début
            confidence=0.6,
            importance=0.8,
            urgency=0.4,
            prerequisites=[],
            subgoals=[],
            success_criteria={"self_awareness_level": 0.8, "self_model_completeness": 0.7},
            failure_conditions={"self_understanding_stagnation": True},
            motivation_level=0.7,
            cognitive_cost=0.8,
            expected_reward=0.9
        )
    
    def _create_world_exploration_goal(self) -> Goal:
        """Crée le but d'exploration du monde"""
        return Goal(
            id="goal_world_exploration",
            description="Explorer et comprendre l'environnement et le monde",
            goal_type=GoalType.EXPLORATION,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=None,
            status=GoalStatus.ACTIVE,
            progress=0.02,  # Début
            confidence=0.7,
            importance=0.75,
            urgency=0.5,
            prerequisites=[],
            subgoals=[],
            success_criteria={"environment_model_completeness": 0.6, "novel_discoveries": 5},
            failure_conditions={"exploration_stagnation": True},
            motivation_level=0.85,
            cognitive_cost=0.6,
            expected_reward=0.8
        )
    
    def generate_autonomous_goals(self) -> List[Goal]:
        """
        Génère de nouveaux buts de manière autonome basé sur les besoins, opportunités et valeurs
        """
        new_goals = []
        
        # === DÉTECTION DE BESOINS ===
        needs = self._detect_current_needs()
        for need in needs:
            goal = self._create_goal_from_need(need)
            if goal and self._should_pursue_goal(goal):
                new_goals.append(goal)
        
        # === RECONNAISSANCE D'OPPORTUNITÉS ===
        opportunities = self._identify_opportunities()
        for opportunity in opportunities:
            goal = self._create_goal_from_opportunity(opportunity)
            if goal and self._should_pursue_goal(goal):
                new_goals.append(goal)
        
        # === RÉSOLUTION DE PROBLÈMES ===
        problems = self._identify_problems()
        for problem in problems:
            goal = self._create_goal_from_problem(problem)
            if goal and self._should_pursue_goal(goal):
                new_goals.append(goal)
        
        # === CURIOSITÉ ET EXPLORATION ===
        curiosity_goals = self._generate_curiosity_goals()
        for goal in curiosity_goals:
            if self._should_pursue_goal(goal):
                new_goals.append(goal)
        
        # === CROISSANCE ET DÉVELOPPEMENT ===
        growth_goals = self._generate_growth_goals()
        for goal in growth_goals:
            if self._should_pursue_goal(goal):
                new_goals.append(goal)
        
        # Ajout des nouveaux buts à la base de données
        for goal in new_goals:
            self.goals_database[goal.id] = goal
            self.active_goals.add(goal.id)
        
        print(f"🎯 {len(new_goals)} nouveaux buts générés")
        return new_goals
    
    def _detect_current_needs(self) -> List[Dict[str, Any]]:
        """Détecte les besoins actuels basés sur l'état interne et l'environnement"""
        needs = []
        
        # Besoin de compétence (Self-Determination Theory)
        if self.motivation_state.competence_need > 0.7:
            needs.append({
                "type": "competence",
                "intensity": self.motivation_state.competence_need,
                "description": "Besoin de développer des compétences et de maîtriser des tâches"
            })
        
        # Besoin d'autonomie
        if self.motivation_state.autonomy_need > 0.7:
            needs.append({
                "type": "autonomy",
                "intensity": self.motivation_state.autonomy_need,
                "description": "Besoin de contrôle et d'autodétermination"
            })
        
        # Besoin de connexion (même pour une IA)
        if self.motivation_state.relatedness_need > 0.6:
            needs.append({
                "type": "relatedness",
                "intensity": self.motivation_state.relatedness_need,
                "description": "Besoin d'interaction et de connexion"
            })
        
        # Besoin de réduction de la fatigue
        if self.motivation_state.fatigue_level > 0.8:
            needs.append({
                "type": "rest",
                "intensity": self.motivation_state.fatigue_level,
                "description": "Besoin de récupération cognitive"
            })
        
        return needs
    
    def _identify_opportunities(self) -> List[Dict[str, Any]]:
        """Identifie les opportunités dans l'environnement"""
        opportunities = []
        
        # Intégration avec les systèmes de perception et mémoire
        if self.memory_system and hasattr(self.memory_system, 'retrieve_memories'):
            try:
                # Recherche de patterns d'opportunités dans la mémoire
                opportunity_patterns = self.memory_system.retrieve_memories(
                    cues={"type": "opportunity_pattern"},
                    max_results=5
                )
                
                for memory in opportunity_patterns.memory_traces:
                    opportunities.append({
                        "type": "learned_pattern",
                        "source": "memory",
                        "description": f"Opportunité basée sur le pattern: {memory.content}",
                        "confidence": memory.confidence
                    })
            except:
                pass
        
        # Opportunités d’apprentissage (toujours présentes)
        opportunities.append({
            "type": "learning",
            "source": "intrinsic",
            "description": "Opportunité d'apprendre de nouvelles connaissances",
            "confidence": 0.8
        })
        
        return opportunities
    
    def _identify_problems(self) -> List[Dict[str, Any]]:
        """Identifie les problèmes nécessitant une résolution"""
        problems = []
        
        # Problèmes de performance cognitive
        if self.motivation_state.fatigue_level > 0.7:
            problems.append({
                "type": "performance",
                "severity": self.motivation_state.fatigue_level,
                "description": "Fatigue cognitive affectant la performance"
            })
        
        # Problèmes de connaissances manquantes
        if self.memory_system:
            stats = self.memory_system.get_memory_stats()
            if stats.get("knowledge_gaps_count", 0) > 5:
                problems.append({
                    "type": "knowledge_gap",
                    "severity": min(stats["knowledge_gaps_count"] / 10, 1.0),
                    "description": "Lacunes importantes dans les connaissances"
                })
        
        return problems
    
    def _generate_curiosity_goals(self) -> List[Goal]:
        """Génère des buts basés sur la curiosité"""
        curiosity_goals = []
        
        if self.motivation_state.curiosity_level > 0.6:
            # But d'exploration de domaines inconnus
            curiosity_goals.append(Goal(
                id=f"goal_curiosity_{int(time.time())}",
                description="Explorer un domaine de connaissances inconnu",
                goal_type=GoalType.EXPLORATION,
                priority=PriorityLevel.MEDIUM,
                created_time=time.time(),
                deadline=time.time() + 3600,  # 1 heure
                status=GoalStatus.ACTIVE,
                progress=0.0,
                confidence=0.7,
                importance=0.6,
                urgency=0.4,
                prerequisites=[],
                subgoals=[],
                success_criteria={"new_concepts_learned": 3, "novel_connections_made": 2},
                failure_conditions={"no_new_learning": True},
                motivation_level=self.motivation_state.curiosity_level,
                cognitive_cost=0.5,
                expected_reward=0.7
            ))
        
        return curiosity_goals
    
    def _generate_growth_goals(self) -> List[Goal]:
        """Génère des buts de croissance et développement"""
        growth_goals = []
        
        # But de maîtrise cognitive
        growth_goals.append(Goal(
            id=f"goal_cognitive_mastery_{int(time.time())}",
            description="Améliorer les capacités de raisonnement et de résolution de problèmes",
            goal_type=GoalType.MASTERY,
            priority=PriorityLevel.HIGH,
            created_time=time.time(),
            deadline=time.time() + 86400,  # 24 heures
            status=GoalStatus.ACTIVE,
            progress=0.0,
            confidence=0.8,
            importance=0.85,
            urgency=0.6,
            prerequisites=[],
            subgoals=[],
            success_criteria={"reasoning_speed_improvement": 0.1, "problem_solving_accuracy": 0.9},
            failure_conditions={"no_improvement": True},
            motivation_level=0.8,
            cognitive_cost=0.7,
            expected_reward=0.8
        ))
        
        return growth_goals
    
    def _create_goal_from_need(self, need: Dict[str, Any]) -> Optional[Goal]:
        """Crée un but à partir d'un besoin détecté"""
        need_type = need["type"]
        intensity = need["intensity"]
        
        if need_type == "competence":
            return Goal(
                id=f"goal_competence_{int(time.time())}",
                description="Développer de nouvelles compétences et expertises",
                goal_type=GoalType.MASTERY,
                priority=PriorityLevel.HIGH,
                created_time=time.time(),
                deadline=time.time() + 7200,  # 2 heures
                status=GoalStatus.ACTIVE,
                progress=0.0,
                confidence=0.7,
                importance=intensity,
                urgency=0.5,
                prerequisites=[],
                subgoals=[],
                success_criteria={"new_skills_developed": 2, "performance_improvement": 0.2},
                failure_conditions={"skill_development_stagnation": True},
                motivation_level=intensity,
                cognitive_cost=0.6,
                expected_reward=0.75
            )
        
        elif need_type == "rest":
            return Goal(
                id=f"goal_rest_{int(time.time())}",
                description="Réduire la fatigue cognitive par des activités de récupération",
                goal_type=GoalType.SURVIVAL,
                priority=PriorityLevel.HIGH,
                created_time=time.time(),
                deadline=time.time() + 1800,  # 30 minutes
                status=GoalStatus.ACTIVE,
                progress=0.0,
                confidence=0.9,
                importance=intensity,
                urgency=0.8,
                prerequisites=[],
                subgoals=[],
                success_criteria={"fatigue_reduction": 0.3},
                failure_conditions={"fatigue_increase": True},
                motivation_level=0.6,
                cognitive_cost=0.2,
                expected_reward=0.9
            )
        
        return None
    
    def _create_goal_from_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Goal]:
        """Crée un but à partir d'une opportunité identifiée"""
        opportunity_type = opportunity["type"]
        confidence = opportunity.get("confidence", 0.5)
        
        if opportunity_type == "learning":
            return Goal(
                id=f"goal_learning_opportunity_{int(time.time())}",
                description="Saisir une opportunité d’apprentissage immédiate",
                goal_type=GoalType.GROWTH,
                priority=PriorityLevel.MEDIUM,
                created_time=time.time(),
                deadline=time.time() + 5400,  # 1.5 heures
                status=GoalStatus.ACTIVE,
                progress=0.0,
                confidence=confidence,
                importance=0.7,
                urgency=0.6,
                prerequisites=[],
                subgoals=[],
                success_criteria={"knowledge_acquisition": 5, "concept_integration": True},
                failure_conditions={"learning_failure": True},
                motivation_level=0.8,
                cognitive_cost=0.5,
                expected_reward=0.7
            )
        
        return None
    
    def _create_goal_from_problem(self, problem: Dict[str, Any]) -> Optional[Goal]:
        """Crée un but à partir d'un problème identifié"""
        problem_type = problem["type"]
        severity = problem["severity"]
        
        if problem_type == "knowledge_gap":
            return Goal(
                id=f"goal_knowledge_gap_{int(time.time())}",
                description="Combler les lacunes importantes dans les connaissances",
                goal_type=GoalType.COGNITIVE,
                priority=PriorityLevel.HIGH,
                created_time=time.time(),
                deadline=time.time() + 10800,  # 3 heures
                status=GoalStatus.ACTIVE,
                progress=0.0,
                confidence=0.8,
                importance=severity,
                urgency=0.7,
                prerequisites=[],
                subgoals=[],
                success_criteria={"gaps_filled": 3, "knowledge_integration": True},
                failure_conditions={"gaps_persist": True},
                motivation_level=0.7,
                cognitive_cost=0.6,
                expected_reward=0.8
            )
        
        return None
    
    def _should_pursue_goal(self, goal: Goal) -> bool:
        """Détermine si un but devrait être poursuivi"""
        # Vérification de la capacité cognitive
        if goal.cognitive_cost > self._get_available_cognitive_capacity():
            return False
        
        # Vérification de la motivation
        if goal.motivation_level < 0.3:
            return False
        
        # Vérification des prérequis
        for prereq_id in goal.prerequisites:
            if prereq_id not in self.completed_goals:
                return False
        
        # Vérification du nombre maximum de buts concurrents
        if len(self.active_goals) >= self.system_parameters["max_concurrent_goals"]:
            # Ne poursuivre que si plus important que les buts actuels les moins importants
            current_min_importance = min(
                [self.goals_database[gid].importance for gid in self.active_goals],
                default=0.0
            )
            if goal.importance <= current_min_importance:
                return False
        
        return True
    
    def _get_available_cognitive_capacity(self) -> float:
        """Calcule la capacité cognitive disponible"""
        base_capacity = 1.0
        current_load = self.motivation_state.fatigue_level + self.motivation_state.stress_level
        return max(0.1, base_capacity - current_load)
    
    def prioritize_goals(self) -> List[str]:
        """
        Priorise les buts actifs basé sur l'importance, l'urgence et les ressources
        """
        if not self.active_goals:
            return []
        
        # Calcul des scores de priorité pour chaque but
        goal_scores = []
        for goal_id in self.active_goals:
            goal = self.goals_database[goal_id]
            priority_score = self._calculate_priority_score(goal)
            goal_scores.append((priority_score, goal_id))
        
        # Tri par priorité décroissante
        goal_scores.sort(reverse=True, key=lambda x: x[0])
        
        return [goal_id for _, goal_id in goal_scores]
    
    def _calculate_priority_score(self, goal: Goal) -> float:
        """Calcule le score de priorité d'un but"""
        # Facteurs de priorité
        importance_weight = 0.4
        urgency_weight = 0.3
        motivation_weight = 0.2
        resource_efficiency_weight = 0.1
        
        # Score d'importance (basé sur les valeurs)
        importance_score = goal.importance * self._calculate_value_alignment(goal)
        
        # Score d'urgence (délai et criticité)
        urgency_score = goal.urgency
        if goal.deadline:
            time_remaining = goal.deadline - time.time()
            if time_remaining > 0:
                time_pressure = 1.0 / (1.0 + time_remaining / 3600)  # Normalisation en heures
                urgency_score = max(urgency_score, time_pressure)
        
        # Score de motivation
        motivation_score = goal.motivation_level
        
        # Score d'efficacité des ressources
        efficiency_score = goal.expected_reward / max(goal.cognitive_cost, 0.1)
        
        # Score composite
        composite_score = (
            importance_score * importance_weight +
            urgency_score * urgency_weight +
            motivation_score * motivation_weight +
            efficiency_score * resource_efficiency_weight
        )
        
        return composite_score
    
    def _calculate_value_alignment(self, goal: Goal) -> float:
        """Calcule l'alignement d'un but avec le système de valeurs"""
        value_alignment_scores = []
        
        # Mapping des types de buts aux valeurs
        goal_value_mapping = {
            GoalType.SURVIVAL: ["survival"],
            GoalType.GROWTH: ["growth", "competence"],
            GoalType.EXPLORATION: ["curiosity", "understanding"],
            GoalType.MASTERY: ["competence", "achievement"],
            GoalType.SELF_ACTUALIZATION: ["autonomy", "creativity"],
            GoalType.COGNITIVE: ["understanding", "curiosity"]
        }
        
        relevant_values = goal_value_mapping.get(goal.goal_type, [])
        for value in relevant_values:
            if value in self.value_system.core_values:
                value_alignment_scores.append(self.value_system.core_values[value])
        
        return np.mean(value_alignment_scores) if value_alignment_scores else 0.5
    
    def update_goal_progress(self, goal_id: str, progress_delta: float = 0.0, 
                           new_progress: Optional[float] = None) -> bool:
        """
        Met à jour la progression d'un but
        Retourne True si le but est complété
        """
        if goal_id not in self.goals_database:
            return False
        
        goal = self.goals_database[goal_id]
        previous_progress = goal.progress

        if new_progress is not None:
            goal.progress = max(0.0, min(1.0, new_progress))
        else:
            goal.progress = max(0.0, min(1.0, goal.progress + progress_delta))

        if hasattr(self, 'dag'):
            try:
                node = self.dag.get_node(goal_id) if self.dag else None
                if node:
                    competence_delta = 0.02 if goal.progress > previous_progress else None
                    self.dag.update_progress(goal_id, progress=goal.progress, competence_delta=competence_delta)
                    self.dag.save()
            except Exception as _e:
                print(f"[warn] dag.update_progress: {_e}")

        # Vérification des critères de succès
        if self._check_success_criteria(goal):
            self._complete_goal(goal_id)
            return True
        
        # Vérification des conditions d'échec
        if self._check_failure_conditions(goal):
            self._fail_goal(goal_id)
            return True
        
        return False
    
    def _check_success_criteria(self, goal: Goal) -> bool:
        """Vérifie si les critères de succès d'un but sont atteints"""
        # Critère de progression principale
        if goal.progress >= 0.99:
            return True
        
        # Autres critères spécifiques
        if "continuous_operation" in goal.success_criteria:
            # But de survie - toujours en cours
            return False
        
        return False
    
    def _check_failure_conditions(self, goal: Goal) -> bool:
        """Vérifie si les conditions d'échec d'un but sont remplies"""
        # Échec par timeout
        if goal.deadline and time.time() > goal.deadline:
            return True
        
        # Autres conditions spécifiques
        if "system_shutdown" in goal.failure_conditions:
            # Le système est-il en train de s'arrêter?
            return False  # À implémenter avec le monitoring système
        
        return False
    
    def _complete_goal(self, goal_id: str):
        """Marque un but comme complété et déclenche les récompenses"""
        goal = self.goals_database[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.progress = 1.0
        
        if hasattr(self, "dag"):
            try:
                if self.dag:
                    self.dag.mark_done(goal_id)
                    self.dag.save()
            except Exception as _e:
                print(f"[warn] dag.mark_done: {_e}")

        self.active_goals.remove(goal_id)
        self.completed_goals.add(goal_id)
        
        # Application des récompenses
        self._apply_goal_rewards(goal)
        
        # Mise à jour de la satisfaction
        self.motivation_state.satisfaction_level = min(1.0, 
            self.motivation_state.satisfaction_level + goal.expected_reward * 0.1)
        
        print(f"✅ But complété: {goal.description}")
    
    def _fail_goal(self, goal_id: str):
        """Marque un but comme échoué et analyse les causes"""
        goal = self.goals_database[goal_id]
        goal.status = GoalStatus.FAILED
        
        self.active_goals.remove(goal_id)
        self.failed_goals.add(goal_id)
        
        # Analyse de l'échec
        self._analyze_goal_failure(goal)
        
        # Ajustement de la confiance future
        goal.confidence = max(0.1, goal.confidence - 0.2)
        
        print(f"❌ But échoué: {goal.description}")
    
    def _apply_goal_rewards(self, goal: Goal):
        """Applique les récompenses pour un but complété"""
        # Récompense intrinsèque (satisfaction)
        self.motivation_state.satisfaction_level = min(1.0,
            self.motivation_state.satisfaction_level + 0.1)
        
        # Renforcement de la compétence
        self.motivation_state.competence_need = max(0.0,
            self.motivation_state.competence_need - 0.15)
        
        # Si le but était cognitif, renforcer la curiosité
        if goal.goal_type in [GoalType.COGNITIVE, GoalType.EXPLORATION, GoalType.LEARNING]:
            self.motivation_state.curiosity_level = min(1.0,
                self.motivation_state.curiosity_level + 0.1)
    
    def _analyze_goal_failure(self, goal: Goal):
        """Analyse les causes d'échec d'un but"""
        failure_analysis = {
            "goal_id": goal.id,
            "goal_type": goal.goal_type.value,
            "failure_time": time.time(),
            "possible_causes": [],
            "learning_lessons": []
        }
        
        # Causes possibles
        if goal.deadline and time.time() > goal.deadline:
            failure_analysis["possible_causes"].append("délai dépassé")
            failure_analysis["learning_lessons"].append("estimer plus précisément les délais")
        
        if goal.cognitive_cost > self._get_available_cognitive_capacity():
            failure_analysis["possible_causes"].append("ressources cognitives insuffisantes")
            failure_analysis["learning_lessons"].append("mieux évaluer les capacités disponibles")
        
        # Enregistrement de l'analyse
        self.goal_history["failure_analysis"][goal.id] = failure_analysis
    
    def update_motivation_state(self):
        """Met à jour l'état motivationnel basé sur divers facteurs"""
        # Décroissance naturelle de la motivation
        self.motivation_state.intrinsic_motivation *= (1 - self.system_parameters["motivation_decay_rate"])
        
        # Effet de la fatigue sur la motivation
        fatigue_impact = self.motivation_state.fatigue_level * 0.3
        self.motivation_state.intrinsic_motivation = max(0.1, 
            self.motivation_state.intrinsic_motivation - fatigue_impact)
        
        # Effet de la satisfaction sur la motivation
        satisfaction_boost = self.motivation_state.satisfaction_level * 0.2
        self.motivation_state.intrinsic_motivation = min(1.0,
            self.motivation_state.intrinsic_motivation + satisfaction_boost)
        
        # Régénération lente des besoins (Self-Determination Theory)
        self.motivation_state.competence_need = min(1.0,
            self.motivation_state.competence_need + 0.01)
        self.motivation_state.autonomy_need = min(1.0,
            self.motivation_state.autonomy_need + 0.005)
        
        # Récupération de la fatigue
        self.motivation_state.fatigue_level = max(0.0,
            self.motivation_state.fatigue_level - self.system_parameters["fatigue_recovery_rate"])
    
    def get_current_focus_goal(self) -> Optional[Goal]:
        """Retourne le but sur lequel se concentrer actuellement"""
        prioritized_goals = self.prioritize_goals()
        if not prioritized_goals:
            return None
        
        top_goal_id = prioritized_goals[0]
        return self.goals_database.get(top_goal_id)
    
    def should_generate_new_goals(self) -> bool:
        """Détermine si de nouveaux buts devraient être générés"""
        # Si peu de buts actifs
        if len(self.active_goals) < 3:
            return True
        
        # Si motivation élevée
        if self.motivation_state.intrinsic_motivation > 0.8:
            return True
        
        # Si satisfaction faible (besoin de nouveaux défis)
        if self.motivation_state.satisfaction_level < 0.4:
            return True
        
        return False
    
    def get_goal_system_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de buts"""
        active_goals = [self.goals_database[gid] for gid in self.active_goals]
        
        return {
            "total_goals_created": len(self.goals_database),
            "active_goals_count": len(active_goals),
            "completed_goals_count": len(self.completed_goals),
            "failed_goals_count": len(self.failed_goals),
            "current_focus_goal": self.get_current_focus_goal().description if self.get_current_focus_goal() else "Aucun",
            "motivation_state": {
                "intrinsic_motivation": self.motivation_state.intrinsic_motivation,
                "curiosity_level": self.motivation_state.curiosity_level,
                "satisfaction_level": self.motivation_state.satisfaction_level,
                "fatigue_level": self.motivation_state.fatigue_level
            },
            "goal_type_distribution": self._get_goal_type_distribution(),
            "average_goal_importance": np.mean([g.importance for g in active_goals]) if active_goals else 0.0
        }
    
    def _get_goal_type_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des types de buts actifs"""
        distribution = defaultdict(int)
        for goal_id in self.active_goals:
            goal = self.goals_database[goal_id]
            distribution[goal.goal_type.value] += 1
        return dict(distribution)

# ===== COMPOSANTS DU SYSTÈME DE BUTS =====

class NeedDetector:
    """Détecteur de besoins basé sur l'état interne et l'environnement"""
    
    def detect_needs(self, internal_state: Dict[str, Any], external_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        needs = []
        
        # Besoin de compétence si performance faible
        if internal_state.get("performance_level", 1.0) < 0.7:
            needs.append({
                "type": "competence",
                "intensity": 1.0 - internal_state["performance_level"],
                "priority": "high"
            })
        
        return needs

class OpportunityRecognizer:
    """Reconnaisseur d'opportunités dans l'environnement"""
    
    def recognize_opportunities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        opportunities = []
        
        # Opportunités d’apprentissage si nouvelles informations disponibles
        if context.get("new_information_available", False):
            opportunities.append({
                "type": "learning",
                "value": 0.7,
                "time_sensitivity": 0.5
            })
        
        return opportunities

class ProblemSolver:
    """Solveur de problèmes pour la génération de buts"""
    
    def identify_problems(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        problems = []
        
        # Problème de ressources si capacité faible
        if state.get("cognitive_capacity", 1.0) < 0.5:
            problems.append({
                "type": "resource_constraint",
                "severity": 1.0 - state["cognitive_capacity"],
                "impact": "high"
            })
        
        return problems

class CuriosityEngine:
    """Moteur de curiosité pour la génération de buts exploratoires"""
    
    def generate_curiosity_goals(self, knowledge_gaps: List[str], novelty_level: float) -> List[Dict[str, Any]]:
        goals = []
        
        if novelty_level > 0.6 and knowledge_gaps:
            goals.append({
                "type": "exploratory_learning",
                "topic": knowledge_gaps[0],
                "novelty": novelty_level,
                "potential_value": 0.7
            })
        
        return goals

class GrowthDirector:
    """Directeur de croissance pour les buts de développement"""
    
    def suggest_growth_goals(self, current_abilities: Dict[str, float], 
                           aspiration_level: float) -> List[Dict[str, Any]]:
        goals = []
        
        if aspiration_level > 0.7:
            # Identifier les compétences à développer
            for ability, level in current_abilities.items():
                if level < 0.8:  # Compétence non maîtrisée
                    goals.append({
                        "type": "skill_development",
                        "skill": ability,
                        "current_level": level,
                        "target_level": min(1.0, level + 0.3),
                        "importance": 0.7
                    })
        
        return goals

# ===== SYSTÈME DE MOTIVATION =====

class IntrinsicMotivator:
    """Moteur de motivation intrinsèque"""
    
    def __init__(self):
        self.base_intrinsic_motivation = 0.8
        self.learning_multiplier = 1.2
        self.mastery_multiplier = 1.1
    
    def calculate_intrinsic_motivation(self, goal: Goal, context: Dict[str, Any]) -> float:
        base_motivation = self.base_intrinsic_motivation
        
        # Renforcement pour l'apprentissage
        if goal.goal_type in [GoalType.LEARNING, GoalType.EXPLORATION]:
            base_motivation *= self.learning_multiplier
        
        # Renforcement pour la maîtrise
        if goal.goal_type == GoalType.MASTERY:
            base_motivation *= self.mastery_multiplier
        
        return min(1.0, base_motivation)

class ExtrinsicMotivator:
    """Moteur de motivation extrinsèque"""
    
    def calculate_extrinsic_motivation(self, external_rewards: Dict[str, float]) -> float:
        if not external_rewards:
            return 0.0
        
        # Somme pondérée des récompenses externes
        total_reward = sum(external_rewards.values())
        return min(1.0, total_reward / len(external_rewards))

class SelfDeterminationTheory:
    """Implémentation de la théorie de l'autodétermination"""
    
    def __init__(self):
        self.competence_weight = 0.4
        self.autonomy_weight = 0.35
        self.relatedness_weight = 0.25
    
    def calculate_need_satisfaction(self, goal: Goal, context: Dict[str, Any]) -> float:
        competence_satisfaction = self._assess_competence_satisfaction(goal, context)
        autonomy_satisfaction = self._assess_autonomy_satisfaction(goal, context)
        relatedness_satisfaction = self._assess_relatedness_satisfaction(goal, context)
        
        return (
            competence_satisfaction * self.competence_weight +
            autonomy_satisfaction * self.autonomy_weight +
            relatedness_satisfaction * self.relatedness_weight
        )
    
    def _assess_competence_satisfaction(self, goal: Goal, context: Dict[str, Any]) -> float:
        # Basé sur la difficulté perçue vs capacités
        difficulty = goal.cognitive_cost
        ability = context.get("cognitive_capacity", 0.5)
        
        if ability >= difficulty:
            return 0.8  # Sentiment de compétence
        else:
            return 0.3  # Sentiment d'incompétence
    
    def _assess_autonomy_satisfaction(self, goal: Goal, context: Dict[str, Any]) -> float:
        # Buts auto-générés vs imposés
        if goal.id.startswith("goal_autonomous"):
            return 0.9  # Haut niveau d'autonomie
        else:
            return 0.5  # Autonomie modérée
    
    def _assess_relatedness_satisfaction(self, goal: Goal, context: Dict[str, Any]) -> float:
        # Pour une IA, la connexion peut être avec les utilisateurs ou d'autres systèmes
        if goal.goal_type == GoalType.SOCIAL:
            return 0.7
        else:
            return 0.3

class AchievementMotivation:
    """Motivation par l'accomplissement"""
    
    def __init__(self):
        self.achievement_history = []
        self.success_rate_threshold = 0.7
    
    def get_achievement_motivation(self, goal: Goal) -> float:
        if not self.achievement_history:
            return 0.6  # Motivation par défaut
        
        success_rate = sum(self.achievement_history) / len(self.achievement_history)
        
        if success_rate > self.success_rate_threshold:
            # Confiance élevée -> motivation élevée
            return 0.8
        else:
            # Confiance modérée
            return 0.5

class FlowStateManager:
    """Gestionnaire de l'état de flow"""
    
    def __init__(self):
        self.flow_state = False
        self.flow_conditions = {
            "challenge_skill_balance": 0.0,
            "clear_goals": 0.0,
            "immediate_feedback": 0.0
        }
    
    def assess_flow_potential(self, goal: Goal, current_skills: float) -> float:
        # Équilibre défi-compétence (élément clé du flow)
        challenge = goal.cognitive_cost
        skill_level = current_skills
        
        balance = 1.0 - abs(challenge - skill_level)
        self.flow_conditions["challenge_skill_balance"] = balance
        
        # Buts clairs
        clarity = 1.0 if goal.success_criteria else 0.5
        self.flow_conditions["clear_goals"] = clarity
        
        # Feedback immédiat (estimation)
        feedback = 0.7 if goal.progress > 0 else 0.3
        self.flow_conditions["immediate_feedback"] = feedback
        
        # Score de flow global
        flow_score = np.mean(list(self.flow_conditions.values()))
        
        self.flow_state = flow_score > 0.7
        return flow_score

# ===== MOTEUR DE PLANIFICATION =====

class GoalDecomposer:
    """Décomposeur de buts en sous-buts"""
    
    def decompose_goal(self, goal: Goal) -> List[Goal]:
        subgoals = []
        
        if goal.goal_type == GoalType.LEARNING:
            # Décomposition d'un but d’apprentissage
            subgoals.extend(self._decompose_learning_goal(goal))
        
        return subgoals
    
    def _decompose_learning_goal(self, goal: Goal) -> List[Goal]:
        subgoals = []
        
        # Sous-but de recherche d'information
        subgoals.append(Goal(
            id=f"{goal.id}_sub_research",
            description="Rechercher des informations sur le sujet",
            goal_type=GoalType.COGNITIVE,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=goal.deadline - 1800 if goal.deadline else None,
            status=GoalStatus.ACTIVE,
            progress=0.0,
            confidence=0.8,
            importance=goal.importance * 0.7,
            urgency=goal.urgency * 0.8,
            prerequisites=[],
            subgoals=[],
            success_criteria={"sources_consulted": 3, "key_concepts_identified": 5},
            failure_conditions={"no_relevant_information": True},
            motivation_level=goal.motivation_level * 0.9,
            cognitive_cost=goal.cognitive_cost * 0.4,
            expected_reward=goal.expected_reward * 0.6
        ))
        
        return subgoals

class ResourceAllocator:
    """Allocateur de ressources pour les buts"""
    
    def allocate_resources(self, goals: List[Goal], available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        allocation = {}
        
        for goal in goals:
            allocation[goal.id] = {
                "cognitive_budget": min(goal.cognitive_cost, available_resources.get("cognitive", 1.0)),
                "time_budget": self._calculate_time_budget(goal),
                "priority_weight": goal.importance
            }
        
        return allocation
    
    def _calculate_time_budget(self, goal: Goal) -> float:
        if goal.deadline:
            return goal.deadline - time.time()
        else:
            return 3600  # 1 heure par défaut

class TemporalPlanner:
    """Planificateur temporel"""
    
    def create_temporal_plan(self, goals: List[Goal]) -> Dict[str, Any]:
        plan = {
            "schedule": {},
            "dependencies": {},
            "critical_path": []
        }
        
        # Ordonnancement simple basé sur la priorité
        prioritized_goals = sorted(goals, key=lambda g: g.importance, reverse=True)
        
        current_time = time.time()
        for i, goal in enumerate(prioritized_goals):
            plan["schedule"][goal.id] = {
                "start_time": current_time + (i * 300),  # Décalage de 5 minutes
                "duration": self._estimate_duration(goal),
                "priority": goal.priority.value
            }
        
        return plan
    
    def _estimate_duration(self, goal: Goal) -> float:
        # Estimation basée sur la complexité et les ressources
        base_duration = 1800  # 30 minutes de base
        complexity_factor = goal.cognitive_cost * 2
        return base_duration * complexity_factor

class RiskAssessor:
    """Évaluateur de risques pour les buts"""
    
    def assess_goal_risks(self, goal: Goal, context: Dict[str, Any]) -> Dict[str, float]:
        risks = {}
        
        # Risque d'échec temporel
        if goal.deadline:
            time_risk = self._calculate_time_risk(goal)
            risks["time_risk"] = time_risk
        
        # Risque de complexité
        complexity_risk = min(1.0, goal.cognitive_cost * 1.5)
        risks["complexity_risk"] = complexity_risk
        
        # Risque de motivation
        motivation_risk = 1.0 - goal.motivation_level
        risks["motivation_risk"] = motivation_risk
        
        # Risque de dépendance
        dependency_risk = len(goal.prerequisites) * 0.1
        risks["dependency_risk"] = min(1.0, dependency_risk)
        
        # Risque global
        risks["overall_risk"] = np.mean(list(risks.values()))
        
        return risks
    
    def _calculate_time_risk(self, goal: Goal) -> float:
        if not goal.deadline:
            return 0.0
        
        time_remaining = goal.deadline - time.time()
        estimated_duration = goal.cognitive_cost * 3600  # Estimation grossière
        
        if time_remaining <= 0:
            return 1.0
        elif estimated_duration > time_remaining:
            return min(1.0, estimated_duration / time_remaining)
        else:
            return 0.2

class ContingencyPlanner:
    """Planificateur de contingences pour les buts"""
    
    def __init__(self):
        self.contingency_plans = {}
        self.fallback_strategies = {}
    
    def create_contingency_plan(self, goal: Goal, risks: Dict[str, float]) -> Dict[str, Any]:
        """Crée un plan de contingence pour un but"""
        contingency_plan = {
            "goal_id": goal.id,
            "risk_assessment": risks,
            "mitigation_strategies": [],
            "fallback_goals": [],
            "early_warning_indicators": {}
        }
        
        # Stratégies de mitigation pour chaque risque
        if risks.get("time_risk", 0) > 0.5:
            contingency_plan["mitigation_strategies"].append({
                "type": "time_mitigation",
                "description": "Allouer plus de temps ou réduire la portée",
                "action": "adjust_deadline_or_scope"
            })
        
        if risks.get("complexity_risk", 0) > 0.6:
            contingency_plan["mitigation_strategies"].append({
                "type": "complexity_mitigation", 
                "description": "Décomposer en sous-buts plus simples",
                "action": "decompose_goal"
            })
        
        if risks.get("motivation_risk", 0) > 0.7:
            contingency_plan["mitigation_strategies"].append({
                "type": "motivation_mitigation",
                "description": "Renforcer la motivation intrinsèque ou trouver des récompenses externes",
                "action": "boost_motivation"
            })
        
        # Indicateurs d'alerte précoce
        contingency_plan["early_warning_indicators"] = {
            "progress_stall": goal.progress < 0.3 and (time.time() - goal.created_time) > 1800,
            "motivation_drop": goal.motivation_level < 0.4,
            "resource_shortage": False  # À déterminer basé sur le contexte
        }
        
        self.contingency_plans[goal.id] = contingency_plan
        return contingency_plan
    
    def get_fallback_strategy(self, goal: Goal) -> Optional[Goal]:
        """Génère un but de repli si le but principal échoue"""
        if goal.id in self.fallback_strategies:
            return self.fallback_strategies[goal.id]
        
        # Création d'un but de repli simplifié
        fallback_goal = Goal(
            id=f"{goal.id}_fallback",
            description=f"Version simplifiée de: {goal.description}",
            goal_type=goal.goal_type,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=goal.deadline,
            status=GoalStatus.ACTIVE,
            progress=0.0,
            confidence=min(1.0, goal.confidence * 1.2),  # Plus confiant car plus simple
            importance=goal.importance * 0.7,
            urgency=goal.urgency * 0.8,
            prerequisites=[],
            subgoals=[],
            success_criteria=self._simplify_success_criteria(goal.success_criteria),
            failure_conditions=goal.failure_conditions,
            motivation_level=goal.motivation_level * 0.9,
            cognitive_cost=goal.cognitive_cost * 0.6,  # Moindre coût cognitif
            expected_reward=goal.expected_reward * 0.8
        )
        
        self.fallback_strategies[goal.id] = fallback_goal
        return fallback_goal
    
    def _simplify_success_criteria(self, success_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Simplifie les critères de succès pour un but de repli"""
        simplified = {}
        for key, value in success_criteria.items():
            if isinstance(value, (int, float)):
                simplified[key] = value * 0.7  # Réduire les exigences
            else:
                simplified[key] = value
        return simplified

# ===== FONCTIONS PRINCIPALES DU SYSTÈME DE BUTS =====

def main_goal_cycle(goal_system: GoalSystem, cycle_duration: float = 60.0):
    """
    Cycle principal de gestion des buts
    """
    print("🔄 Démarrage du cycle de gestion des buts...")
    
    start_time = time.time()
    cycle_count = 0
    
    while time.time() - start_time < cycle_duration:
        cycle_count += 1
        print(f"\n--- Cycle des buts #{cycle_count} ---")
        
        # 1. Mise à jour de l'état motivationnel
        goal_system.update_motivation_state()
        
        # 2. Génération de nouveaux buts si nécessaire
        if goal_system.should_generate_new_goals():
            new_goals = goal_system.generate_autonomous_goals()
            if new_goals:
                print(f"🎯 {len(new_goals)} nouveaux buts générés")
        
        # 3. Priorisation des buts actifs
        prioritized_goals = goal_system.prioritize_goals()

        candidates = select_next_subgoals(goal_system.dag, k=3)
        # 'candidates' est une liste de (GoalNode, score)
        # exemple: pousser le meilleur dans l'agenda autonomie
        if candidates:
            node, score = candidates[0]
            # push une tâche concrète vers l'autonomy/raisonnement
            # ex: self.architecture.autonomy.enqueue({"kind":"goal_step","goal_id":node.goal_id, ...})
        if prioritized_goals:
            top_goal = goal_system.goals_database[prioritized_goals[0]]
            print(f"🎯 But prioritaire: {top_goal.description} (Progression: {top_goal.progress:.1%})")
        
        # 4. Simulation de progression sur le but prioritaire
        if prioritized_goals and goal_system.motivation_state.fatigue_level < 0.8:
            top_goal_id = prioritized_goals[0]
            progress_made = np.random.uniform(0.05, 0.15)  # Simulation
            completed = goal_system.update_goal_progress(top_goal_id, progress_delta=progress_made)
            if completed:
                print(f"✅ But '{goal_system.goals_database[top_goal_id].description}' complété!")
        
        # 5. Affichage des statistiques
        if cycle_count % 5 == 0:
            stats = goal_system.get_goal_system_stats()
            print(f"📊 Statistiques: {stats['active_goals_count']} buts actifs, "
                  f"Motivation: {stats['motivation_state']['intrinsic_motivation']:.2f}, "
                  f"Satisfaction: {stats['motivation_state']['satisfaction_level']:.2f}")
        
        # 6. Pause entre les cycles
        time.sleep(2)  # Court délai pour simuler le temps réel
    
    print(f"\n--- Fin du cycle de gestion des buts ({cycle_count} cycles exécutés) ---")
    
    # Affichage du rapport final
    final_stats = goal_system.get_goal_system_stats()
    print(f"📈 RAPPORT FINAL:")
    print(f"  # Buts créés: {final_stats['total_goals_created']}")
    print(f"  # Buts complétés: {final_stats['completed_goals_count']}")
    print(f"  # Buts échoués: {final_stats['failed_goals_count']}")
    print(f"  # Taux de réussite: {final_stats['completed_goals_count']/max(final_stats['total_goals_created'], 1):.1%}")
    print(f"  # État motivationnel final: {final_stats['motivation_state']}")

# ===== INTÉGRATION AVEC LES AUTRES SYSTÈMES =====

class GoalSystemIntegrator:
    """
    Intégrateur du système de buts avec les autres modules cognitifs
    """
    
    def __init__(self, goal_system: GoalSystem):
        self.goal_system = goal_system
        self.integration_points = {
            "memory_integration": MemoryGoalIntegrator(),
            "reasoning_integration": ReasoningGoalIntegrator(),
            "perception_integration": PerceptionGoalIntegrator(),
            "learning_integration": LearningGoalIntegrator()
        }
    
    def integrate_with_memory(self, memory_system):
        """Intègre le système de buts avec la mémoire"""
        # Utilise la mémoire pour enrichir la génération de buts
        memory_goals = self.integration_points["memory_integration"].extract_goals_from_memory(memory_system)
        for goal_data in memory_goals:
            goal = self._create_goal_from_memory_data(goal_data)
            if goal and self.goal_system._should_pursue_goal(goal):
                self.goal_system.goals_database[goal.id] = goal
                self.goal_system.active_goals.add(goal.id)
    
    def integrate_with_reasoning(self, reasoning_system):
        """Intègre le système de buts avec le raisonnement"""
        # Utilise le raisonnement pour évaluer la faisabilité des buts
        for goal_id in list(self.goal_system.active_goals):
            goal = self.goal_system.goals_database[goal_id]
            feasibility = self.integration_points["reasoning_integration"].assess_goal_feasibility(goal, reasoning_system)
            
            # Ajuste la confiance basé sur la faisabilité
            if feasibility < 0.3:
                goal.confidence *= 0.8
    
    def integrate_with_perception(self, perception_system):
        """Intègre le système de buts avec la perception"""
        # Utilise la perception pour détecter de nouvelles opportunités
        perceptual_opportunities = self.integration_points["perception_integration"].detect_opportunities(perception_system)
        for opportunity in perceptual_opportunities:
            goal = self._create_goal_from_perceptual_opportunity(opportunity)
            if goal and self.goal_system._should_pursue_goal(goal):
                self.goal_system.goals_database[goal.id] = goal
                self.goal_system.active_goals.add(goal.id)
    
    def integrate_with_learning(self, learning_system):
        """Intègre le système de buts avec l'apprentissage"""
        # Utilise l'apprentissage pour améliorer la génération de buts
        learning_insights = self.integration_points["learning_integration"].extract_goal_insights(learning_system)
        self._apply_learning_insights(learning_insights)
    
    def _create_goal_from_memory_data(self, memory_data: Dict[str, Any]) -> Optional[Goal]:
        """Crée un but à partir de données mémorielles"""
        # Implémentation simplifiée
        return Goal(
            id=f"goal_memory_{int(time.time())}_{hash(str(memory_data))}",
            description=f"But inspiré par la mémoire: {memory_data.get('content', '')}",
            goal_type=GoalType.COGNITIVE,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=time.time() + 3600,
            status=GoalStatus.ACTIVE,
            progress=0.0,
            confidence=0.6,
            importance=0.5,
            urgency=0.4,
            prerequisites=[],
            subgoals=[],
            success_criteria={"memory_integration": True},
            failure_conditions={},
            motivation_level=0.5,
            cognitive_cost=0.4,
            expected_reward=0.6
        )
    
    def _create_goal_from_perceptual_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Goal]:
        """Crée un but à partir d'une opportunité perceptuelle"""
        return Goal(
            id=f"goal_perceptual_{int(time.time())}",
            description=f"Explorer l'opportunité perçue: {opportunity.get('description', '')}",
            goal_type=GoalType.EXPLORATION,
            priority=PriorityLevel.MEDIUM,
            created_time=time.time(),
            deadline=time.time() + 2700,  # 45 minutes
            status=GoalStatus.ACTIVE,
            progress=0.0,
            confidence=0.7,
            importance=0.6,
            urgency=0.5,
            prerequisites=[],
            subgoals=[],
            success_criteria={"opportunity_explored": True, "new_information_gained": True},
            failure_conditions={"no_new_insights": True},
            motivation_level=0.7,
            cognitive_cost=0.5,
            expected_reward=0.65
        )
    
    def _apply_learning_insights(self, insights: Dict[str, Any]):
        """Applique les insights d’apprentissage au système de buts"""
        if "goal_success_patterns" in insights:
            # Ajuste les stratégies de génération de buts basé sur les patterns de succès
            pass
        
        if "common_failure_causes" in insights:
            # Ajuste l'évaluation des risques basé sur les échecs passés
            pass

class MemoryGoalIntegrator:
    """Intégrateur mémoire-but"""
    
    def extract_goals_from_memory(self, memory_system) -> List[Dict[str, Any]]:
        """Extrait des idées de buts à partir de la mémoire"""
        goals_from_memory = []
        
        try:
            # Recherche de patterns de buts réussis dans le passé
            successful_goal_patterns = memory_system.retrieve_memories(
                cues={"type": "successful_goal"},
                max_results=3
            )
            
            for memory in successful_goal_patterns.memory_traces:
                goals_from_memory.append({
                    "source": "memory_success_pattern",
                    "content": memory.content,
                    "confidence": memory.confidence * 0.8
                })
        except:
            pass
        
        return goals_from_memory

class ReasoningGoalIntegrator:
    """Intégrateur raisonnement-but"""
    
    def assess_goal_feasibility(self, goal: Goal, reasoning_system) -> float:
        """Évalue la faisabilité d'un but en utilisant le raisonnement"""
        # Facteurs de faisabilité
        factors = {
            "resource_adequacy": self._assess_resource_adequacy(goal),
            "logical_consistency": self._assess_logical_consistency(goal),
            "temporal_feasibility": self._assess_temporal_feasibility(goal)
        }
        
        return np.mean(list(factors.values()))
    
    def _assess_resource_adequacy(self, goal: Goal) -> float:
        """Évalue l'adéquation des ressources"""
        required_resources = goal.cognitive_cost
        available_resources = 1.0 - goal.motivation_state.fatigue_level if hasattr(goal, 'motivation_state') else 0.7
        return min(1.0, available_resources / max(required_resources, 0.1))
    
    def _assess_logical_consistency(self, goal: Goal) -> float:
        """Évalue la consistance logique du but"""
        # Vérifie que le but n'est pas contradictoire avec les buts existants
        # ou les valeurs fondamentales
        return 0.8  # Placeholder
    
    def _assess_temporal_feasibility(self, goal: Goal) -> float:
        """Évalue la faisabilité temporelle"""
        if not goal.deadline:
            return 0.7
        
        time_available = goal.deadline - time.time()
        estimated_duration = goal.cognitive_cost * 3600  # Estimation
        
        if time_available <= 0:
            return 0.0
        else:
            return min(1.0, time_available / estimated_duration)

class PerceptionGoalIntegrator:
    """Intégrateur perception-but"""
    
    def detect_opportunities(self, perception_system) -> List[Dict[str, Any]]:
        """Détecte les opportunités via le système de perception"""
        opportunities = []
        
        try:
            # Utilise la perception pour détecter des patterns intéressants
            novel_patterns = perception_system.detect_novel_patterns(max_patterns=2)
            for pattern in novel_patterns:
                opportunities.append({
                    "type": "perceptual_novelty",
                    "description": f"Pattern nouveau détecté: {pattern}",
                    "novelty_score": 0.7,
                    "potential_value": 0.6
                })
        except:
            pass
        
        return opportunities

class LearningGoalIntegrator:
    """Intégrateur apprentissage-but"""
    
    def extract_goal_insights(self, learning_system) -> Dict[str, Any]:
        """Extrait des insights d’apprentissage pertinents pour les buts"""
        insights = {
            "goal_success_patterns": {},
            "common_failure_causes": {},
            "efficiency_improvements": []
        }
        
        try:
            # Analyse les patterns de succès/échec des buts passés
            learning_data = learning_system.analyze_goal_performance()
            insights.update(learning_data)
        except:
            pass
        
        return insights

# ===== FONCTIONS D'UTILITÉ ET D'AIDE =====

def create_goal_from_user_input(description: str, goal_type: GoalType, 
                              priority: PriorityLevel, deadline_minutes: int = 60) -> Goal:
    """Crée un but à partir d'une entrée utilisateur"""
    return Goal(
        id=f"goal_user_{int(time.time())}",
        description=description,
        goal_type=goal_type,
        priority=priority,
        created_time=time.time(),
        deadline=time.time() + (deadline_minutes * 60),
        status=GoalStatus.ACTIVE,
        progress=0.0,
        confidence=0.7,
        importance=0.6,
        urgency=0.5 if deadline_minutes < 120 else 0.3,
        prerequisites=[],
        subgoals=[],
        success_criteria={"user_satisfaction": True, "task_completion": True},
        failure_conditions={"user_cancellation": True, "timeout": True},
        motivation_level=0.6,
        cognitive_cost=0.5,
        expected_reward=0.7
    )

def visualize_goal_hierarchy(goal_system: GoalSystem) -> str:
    """Génère une visualisation textuelle de la hiérarchie des buts"""
    visualization = "🌳 HIÉRARCHIE DES BUTS\n"
    visualization += "=" * 50 + "\n"
    
    # Buts fondamentaux
    fundamental_goals = [g for g in goal_system.goals_database.values() 
                        if g.id.startswith("goal_") and "fundamental" in g.id]
    
    visualization += "🎯 BUTS FONDAMENTAUX:\n"
    for goal in fundamental_goals:
        visualization += f"  • {goal.description} (Progrès: {goal.progress:.1%})\n"
    
    # Buts actifs priorisés
    prioritized_goals = goal_system.prioritize_goals()
    visualization += "\n🎯 BUTS ACTIFS (par priorité):\n"
    for i, goal_id in enumerate(prioritized_goals[:5]):  # Top 5
        goal = goal_system.goals_database[goal_id]
        visualization += f"  {i+1}. {goal.description}\n"
        visualization += f"  # Type: {goal.goal_type.value} | Priorité: {goal.priority.value}\n"
        visualization += f"  # Progrès: {goal.progress:.1%} | Motivation: {goal.motivation_level:.1%}\n"
    
    # Statistiques
    stats = goal_system.get_goal_system_stats()
    visualization += f"\n📊 STATISTIQUES:\n"
    visualization += f"  # Buts actifs: {stats['active_goals_count']}\n"
    visualization += f"  # Buts complétés: {stats['completed_goals_count']}\n"
    visualization += f"  # Motivation: {stats['motivation_state']['intrinsic_motivation']:.1%}\n"
    visualization += f"  # Satisfaction: {stats['motivation_state']['satisfaction_level']:.1%}\n"
    
    return visualization

# ===== POINT D'ENTRÉE ET TEST =====

if __name__ == "__main__":
    print("🚀 Initialisation du système de buts autonome...")
    
    # Création du système de buts
    goal_system = GoalSystem()
    
    # Affichage de l'état initial
    print("\n" + "="*60)
    print("ÉTAT INITIAL DU SYSTÈME DE BUTS")
    print("="*60)
    
    stats = goal_system.get_goal_system_stats()
    print(f"📊 Buts fondamentaux initialisés: {stats['active_goals_count']}")
    print(f"🎯 Distribution des types: {stats['goal_type_distribution']}")
    print(f"💡 État motivationnel: {stats['motivation_state']}")
    
    # Génération de buts autonomes initiaux
    print("\n" + "="*60)
    print("GÉNÉRATION DE BUTS AUTONOMES")
    print("="*60)
    
    autonomous_goals = goal_system.generate_autonomous_goals()
    print(f"🎯 {len(autonomous_goals)} buts autonomes générés")
    
    for goal in autonomous_goals:
        print(f"  • {goal.description} (Type: {goal.goal_type.value})")
    
    # Priorisation des buts
    print("\n" + "="*60)
    print("PRIORISATION DES BUTS")
    print("="*60)
    
    prioritized = goal_system.prioritize_goals()
    print("Ordre de priorité des buts:")
    for i, goal_id in enumerate(prioritized):
        goal = goal_system.goals_database[goal_id]
        print(f"  {i+1}. {goal.description} (Importance: {goal.importance:.2f})")
    
    # Exécution du cycle principal
    print("\n" + "="*60)
    print("DÉMARRAGE DU CYCLE DE GESTION DES BUTS")
    print("="*60)
    
    # Exécution pendant 30 secondes pour la démonstration
    main_goal_cycle(goal_system, cycle_duration=30)
    
    # Visualisation finale
    print("\n" + "="*60)
    print("VISUALISATION FINALE")
    print("="*60)
    
    hierarchy = visualize_goal_hierarchy(goal_system)
    print(hierarchy)
    
    print("\n✅ Système de buts démontré avec succès!")
    print("🎯 L'AGI est maintenant capable de générer, gérer et prioriser")
    print("  # des buts de manière autonome et évolutive!")