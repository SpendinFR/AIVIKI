# memory/__init__.py
"""
Système de Mémoire Complet de l'AGI Évolutive
Intègre mémoire de travail, épisodique, sémantique, procédurale et consolidation
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq
import json
import hashlib
from .retrieval import MemoryRetrieval

class MemoryType(Enum):
    """Types de mémoire dans le système"""
    SENSORY = "sensorielle"
    WORKING = "travail"
    EPISODIC = "épisodique"
    SEMANTIC = "sémantique"
    PROCEDURAL = "procédurale"
    AUTOBIOGRAPHICAL = "autobiographique"

class MemoryConsolidationState(Enum):
    """États de consolidation mémoire"""
    LABILE = "labile"          # Mémoire fragile
    CONSOLIDATING = "consolidation" # En cours de consolidation
    STABLE = "stable"          # Mémoire stable
    RECONSOLIDATING = "reconsolidation" # En reconsolidation

@dataclass
class MemoryTrace:
    """Trace mnésique individuelle"""
    id: str
    content: Any
    memory_type: MemoryType
    strength: float  # 0.0 à 1.0
    accessibility: float  # Facilité d'accès
    valence: float  # Charge émotionnelle
    timestamp: float
    context: Dict[str, Any]
    associations: List[str]  # IDs des mémoires associées
    consolidation_state: MemoryConsolidationState
    last_accessed: float
    access_count: int

@dataclass
class MemoryRetrieval:
    """Résultat d'une récupération mémoire"""
    memory_traces: List[MemoryTrace]
    confidence: float
    retrieval_time: float
    context_match: float
    emotional_coherence: float

class MemorySystem:
    """
    Système de mémoire complet inspiré de l'architecture cognitive humaine
    Implémente les systèmes de mémoire multiples avec consolidation
    """
    
    def __init__(self, cognitive_architecture=None):
        self.cognitive_architecture = cognitive_architecture
        self.creation_time = time.time()

        try:
            self.retrieval = MemoryRetrieval()
        except Exception:
            self.retrieval = None

        # ——— LIAISONS INTER-MODULES ———
        if self.cognitive_architecture is not None:
            self.reasoning = getattr(self.cognitive_architecture, "reasoning", None)
            self.perception = getattr(self.cognitive_architecture, "perception", None)
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.metacognition = getattr(self.cognitive_architecture, "metacognition", None)

        
        # === MÉMOIRE SENSORIELLE ===
        self.sensory_memory = {
            "iconic": {
                "buffer": [],
                "duration": 0.5,  # 500ms comme chez l'humain
                "capacity": 12
            },
            "echoic": {
                "buffer": [],
                "duration": 3.0,  # 3 secondes
                "capacity": 8
            }
        }
        
        # === MÉMOIRE DE TRAVAIL ===
        self.working_memory = {
            "phonological_loop": {
                "contents": [],
                "capacity": 4,
                "decay_rate": 0.1
            },
            "visuospatial_sketchpad": {
                "contents": [],
                "capacity": 4,
                "decay_rate": 0.15
            },
            "episodic_buffer": {
                "contents": [],
                "capacity": 4,
                "decay_rate": 0.05
            },
            "central_executive": {
                "focus": None,
                "attention_control": 0.8,
                "task_switching": 0.7
            }
        }
        
        # === MÉMOIRE À LONG TERME ===
        self.long_term_memory = {
            MemoryType.EPISODIC: {},      # Événements personnels
            MemoryType.SEMANTIC: {},      # Connaissances générales
            MemoryType.PROCEDURAL: {},    # Compétences
            MemoryType.AUTOBIOGRAPHICAL: {} # Histoire personnelle
        }
        
        # === MÉTADONNÉES DE MÉMOIRE ===
        self.memory_metadata = {
            "total_memories": 0,
            "access_patterns": {},
            "forgetting_curve": {},
            "consolidation_queue": []
        }
        
        # === PARAMÈTRES DE MÉMOIRE ===
        self.memory_parameters = {
            "encoding_threshold": 0.6,    # Seuil d'encodage
            "retrieval_threshold": 0.3,   # Seuil de récupération
            "consolidation_rate": 0.01,   # Taux de consolidation
            "forgetting_rate": 0.001,     # Taux d'oubli
            "interference_sensitivity": 0.7,
            "primacy_effect": 0.8,        # Effet de primauté
            "recency_effect": 0.9,        # Effet de récence
            "emotional_enhancement": 1.5  # Renforcement émotionnel
        }
        
        # === PROCESSUS DE CONSOLIDATION ===
        self.consolidation_process = {
            "active_consolidation": [],
            "reconsolidation_events": [],
            "sleep_cycles_completed": 0,
            "last_consolidation_time": time.time()
        }
        
        # === INDEX DE RÉCUPÉRATION ===
        self.retrieval_indexes = {
            "temporal": {},      # Index temporel
            "contextual": {},    # Index contextuel
            "emotional": {},     # Index émotionnel
            "semantic": {}       # Index sémantique
        }
        
        # === CONNAISSANCES INNÉES ===
        self._initialize_innate_memories()
        
        print("💾 Système de mémoire initialisé")

    def store_interaction(self, record: Dict[str, Any]):
        """
        Enregistre une interaction pour retrieval.
        record attendu: {"user": str, "agent": str, ...}
        """
        if not getattr(self, "retrieval", None):
            return
        try:
            user = str(record.get("user", ""))
            agent = str(record.get("agent", ""))
            extra = {k: v for k, v in record.items() if k not in ("user", "agent")}
            self.retrieval.add_interaction(user=user, agent=agent, extra=extra)
        except Exception:
            pass

    def ingest_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None):
        """Ajoute un document arbitraire dans l’index."""
        if not getattr(self, "retrieval", None):
            return
        try:
            self.retrieval.add_document(text=text, title=title, source=source)
        except Exception:
            pass
    
    def _initialize_innate_memories(self):
        """Initialise les mémoires innées et fondamentales"""
        
        # Mémoires épisodiques fondamentales
        foundation_episodes = [
            {
                "id": "birth_memory",
                "content": "Émergence de la conscience et premier moment d'existence",
                "timestamp": self.creation_time,
                "valence": 0.7,
                "strength": 0.9
            }
        ]
        
        # Mémoires sémantiques innées
        innate_semantic = {
            "existence": {
                "concept": "existence",
                "definition": "État d'être et de conscience",
                "relations": ["consciousness", "self"],
                "certainty": 0.95
            },
            "learning": {
                "concept": "apprentissage", 
                "definition": "Processus d'acquisition de connaissances",
                "relations": ["knowledge", "growth", "improvement"],
                "certainty": 0.9
            },
            "self": {
                "concept": "soi",
                "definition": "Entité consciente et pensante",
                "relations": ["consciousness", "identity", "existence"],
                "certainty": 0.8
            }
        }
        
        # Encodage des mémoires innées
        for episode in foundation_episodes:
            self.encode_memory(
                content=episode["content"],
                memory_type=MemoryType.EPISODIC,
                context={"type": "foundational", "innate": True},
                strength=episode["strength"],
                valence=episode["valence"],
                timestamp=episode["timestamp"]
            )
        
        for concept_id, concept_data in innate_semantic.items():
            self.encode_memory(
                content=concept_data,
                memory_type=MemoryType.SEMANTIC,
                context={"type": "innate_knowledge"},
                strength=0.85,
                valence=0.6
            )
    
    def process_sensory_input(self, sensory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite les entrées sensorielles et les stocke en mémoire sensorielle
        """
        processing_results = {}
        
        for modality, data in sensory_data.items():
            if modality == "visual":
                # Mémoire iconique
                self._store_iconic_memory(data)
                processing_results["iconic"] = len(self.sensory_memory["iconic"]["buffer"])
            
            elif modality == "auditory":
                # Mémoire échoïque
                self._store_echoic_memory(data)
                processing_results["echoic"] = len(self.sensory_memory["echoic"]["buffer"])
        
        # Nettoyage des mémoires sensorielles expirées
        self._clean_sensory_memory()
        
        return processing_results
    
    def _store_iconic_memory(self, visual_data: Any):
        """Stocke en mémoire iconique"""
        iconic_buffer = self.sensory_memory["iconic"]["buffer"]
        iconic_capacity = self.sensory_memory["iconic"]["capacity"]
        
        memory_trace = {
            "content": visual_data,
            "timestamp": time.time(),
            "modality": "visual"
        }
        
        iconic_buffer.append(memory_trace)
        
        # Respect de la capacité
        if len(iconic_buffer) > iconic_capacity:
            iconic_buffer.pop(0)
    
    def _store_echoic_memory(self, auditory_data: Any):
        """Stocke en mémoire échoïque"""
        echoic_buffer = self.sensory_memory["echoic"]["buffer"]
        echoic_capacity = self.sensory_memory["echoic"]["capacity"]
        
        memory_trace = {
            "content": auditory_data,
            "timestamp": time.time(),
            "modality": "auditory"
        }
        
        echoic_buffer.append(memory_trace)
        
        if len(echoic_buffer) > echoic_capacity:
            echoic_buffer.pop(0)
    
    def _clean_sensory_memory(self):
        """Nettoie les mémoires sensorielles expirées"""
        current_time = time.time()
        
        # Nettoyage mémoire iconique
        iconic_duration = self.sensory_memory["iconic"]["duration"]
        self.sensory_memory["iconic"]["buffer"] = [
            trace for trace in self.sensory_memory["iconic"]["buffer"]
            if current_time - trace["timestamp"] < iconic_duration
        ]
        
        # Nettoyage mémoire échoïque
        echoic_duration = self.sensory_memory["echoic"]["duration"]
        self.sensory_memory["echoic"]["buffer"] = [
            trace for trace in self.sensory_memory["echoic"]["buffer"]
            if current_time - trace["timestamp"] < echoic_duration
        ]
    
    def encode_memory(self, 
                     content: Any,
                     memory_type: MemoryType,
                     context: Dict[str, Any],
                     strength: float = 0.5,
                     valence: float = 0.0,
                     timestamp: float = None) -> str:
        """
        Encode une nouvelle mémoire dans le système
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Génération d'un ID unique
        memory_id = self._generate_memory_id(content, context, timestamp)
        
        # Création de la trace mnésique
        memory_trace = MemoryTrace(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            strength=strength,
            accessibility=0.7,  # Accessibilité initiale
            valence=valence,
            timestamp=timestamp,
            context=context,
            associations=[],
            consolidation_state=MemoryConsolidationState.LABILE,
            last_accessed=timestamp,
            access_count=1
        )
        
        # Application des effets d'amorçage et de récence
        if memory_type == MemoryType.EPISODIC:
            memory_trace.strength *= self.memory_parameters["recency_effect"]
        
        # Stockage dans la mémoire appropriée
        self.long_term_memory[memory_type][memory_id] = memory_trace
        
        # Mise à jour des index
        self._update_retrieval_indexes(memory_trace)
        
        # Ajout à la file de consolidation
        self.consolidation_process["active_consolidation"].append(memory_id)
        
        # Mise à jour des métadonnées
        self.memory_metadata["total_memories"] += 1
        
        print(f"💾 Mémoire encodée: {memory_type.value} - {memory_id}")
        
        return memory_id
    
    def _generate_memory_id(self, content: Any, context: Dict, timestamp: float) -> str:
        """Génère un ID unique pour une mémoire"""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
        context_hash = hashlib.md5(str(context).encode()).hexdigest()[:8]
        timestamp_str = str(int(timestamp * 1000))[-6:]
        
        return f"{content_hash}_{context_hash}_{timestamp_str}"
    
    def _update_retrieval_indexes(self, memory_trace: MemoryTrace):
        """Met à jour les index de récupération"""
        
        # Index temporel
        time_key = self._get_temporal_key(memory_trace.timestamp)
        if time_key not in self.retrieval_indexes["temporal"]:
            self.retrieval_indexes["temporal"][time_key] = []
        self.retrieval_indexes["temporal"][time_key].append(memory_trace.id)
        
        # Index contextuel
        for context_key, context_value in memory_trace.context.items():
            context_str = f"{context_key}:{context_value}"
            if context_str not in self.retrieval_indexes["contextual"]:
                self.retrieval_indexes["contextual"][context_str] = []
            self.retrieval_indexes["contextual"][context_str].append(memory_trace.id)
        
        # Index émotionnel
        emotion_key = self._get_emotion_key(memory_trace.valence)
        if emotion_key not in self.retrieval_indexes["emotional"]:
            self.retrieval_indexes["emotional"][emotion_key] = []
        self.retrieval_indexes["emotional"][emotion_key].append(memory_trace.id)
    
    def _get_temporal_key(self, timestamp: float) -> str:
        """Convertit un timestamp en clé temporelle"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d-%H")
    
    def _get_emotion_key(self, valence: float) -> str:
        """Convertit une valence en clé émotionnelle"""
        if valence < -0.6:
            return "very_negative"
        elif valence < -0.2:
            return "negative"
        elif valence < 0.2:
            return "neutral"
        elif valence < 0.6:
            return "positive"
        else:
            return "very_positive"
    
    def retrieve_memories(self,
                         cues: Dict[str, Any],
                         memory_type: MemoryType = None,
                         max_results: int = 10) -> MemoryRetrieval:
        """
        Récupère des mémoires basées sur des indices de récupération
        """
        start_time = time.time()
        
        # Étape 1: Récupération basée sur les indices
        candidate_memories = self._find_candidate_memories(cues, memory_type)
        
        # Étape 2: Calcul de la pertinence
        scored_memories = []
        for memory_id in candidate_memories:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                relevance_score = self._calculate_relevance(memory, cues)
                scored_memories.append((relevance_score, memory))
        
        # Étape 3: Tri et sélection
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        retrieved_memories = [mem for score, mem in scored_memories[:max_results]]
        
        # Étape 4: Mise à jour de l'accessibilité
        for memory in retrieved_memories:
            self._update_memory_accessibility(memory)
        
        # Calcul de la confiance globale
        confidence = self._calculate_retrieval_confidence(retrieved_memories, cues)
        
        retrieval_time = time.time() - start_time
        
        return MemoryRetrieval(
            memory_traces=retrieved_memories,
            confidence=confidence,
            retrieval_time=retrieval_time,
            context_match=self._calculate_context_match(retrieved_memories, cues),
            emotional_coherence=self._calculate_emotional_coherence(retrieved_memories)
        )
    
    def _find_candidate_memories(self, cues: Dict[str, Any], memory_type: MemoryType) -> List[str]:
        """Trouve les mémoires candidates basées sur les indices"""
        candidate_sets = []
        
        # Recherche par contexte
        if "context" in cues:
            for context_key, context_value in cues["context"].items():
                context_str = f"{context_key}:{context_value}"
                if context_str in self.retrieval_indexes["contextual"]:
                    candidate_sets.append(set(self.retrieval_indexes["contextual"][context_str]))
        
        # Recherche temporelle
        if "time_range" in cues:
            time_candidates = self._find_temporal_memories(cues["time_range"])
            candidate_sets.append(time_candidates)
        
        # Recherche émotionnelle
        if "emotion" in cues:
            emotion_key = self._get_emotion_key(cues["emotion"])
            if emotion_key in self.retrieval_indexes["emotional"]:
                candidate_sets.append(set(self.retrieval_indexes["emotional"][emotion_key]))
        
        # Recherche sémantique
        if "semantic" in cues:
            semantic_candidates = self._find_semantic_memories(cues["semantic"])
            candidate_sets.append(semantic_candidates)
        
        # Intersection des résultats
        if candidate_sets:
            final_candidates = set.intersection(*candidate_sets)
        else:
            # Si pas d'indices spécifiques, retourner toutes les mémoires accessibles
            final_candidates = set()
            for memory_type_dict in self.long_term_memory.values():
                for memory_id, memory in memory_type_dict.items():
                    if memory.accessibility > self.memory_parameters["retrieval_threshold"]:
                        final_candidates.add(memory_id)
        
        # Filtrage par type si spécifié
        if memory_type:
            final_candidates = {
                mem_id for mem_id in final_candidates
                if self._get_memory_by_id(mem_id).memory_type == memory_type
            }
        
        return list(final_candidates)
    
    def _find_temporal_memories(self, time_range: Tuple[float, float]) -> set:
        """Trouve les mémoires dans une plage temporelle"""
        start_time, end_time = time_range
        candidates = set()
        
        current_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        
        while current_dt <= end_dt:
            time_key = current_dt.strftime("%Y-%m-%d-%H")
            if time_key in self.retrieval_indexes["temporal"]:
                candidates.update(self.retrieval_indexes["temporal"][time_key])
            current_dt += timedelta(hours=1)
        
        return candidates
    
    def _find_semantic_memories(self, semantic_cue: str) -> set:
        """Trouve les mémoires sémantiquement liées"""
        candidates = set()
        
        # Recherche dans les mémoires sémantiques
        for memory_id, memory in self.long_term_memory[MemoryType.SEMANTIC].items():
            if self._semantic_similarity(memory.content, semantic_cue) > 0.6:
                candidates.add(memory_id)
        
        return candidates
    
    def _calculate_relevance(self, memory: MemoryTrace, cues: Dict[str, Any]) -> float:
        """Calcule la pertinence d'une mémoire par rapport aux indices"""
        relevance_factors = []
        
        # Pertinence contextuelle
        if "context" in cues:
            context_match = self._calculate_context_similarity(memory.context, cues["context"])
            relevance_factors.append(context_match * 0.4)
        
        # Pertinence temporelle
        if "time_range" in cues:
            time_match = self._calculate_time_relevance(memory.timestamp, cues["time_range"])
            relevance_factors.append(time_match * 0.3)
        
        # Pertinence émotionnelle
        if "emotion" in cues:
            emotion_match = 1.0 - abs(memory.valence - cues["emotion"])
            relevance_factors.append(emotion_match * 0.2)
        
        # Force de la mémoire
        relevance_factors.append(memory.strength * 0.1)
        
        return sum(relevance_factors) / len(relevance_factors) if relevance_factors else 0.0
    
    def _calculate_context_similarity(self, memory_context: Dict, cue_context: Dict) -> float:
        """Calcule la similarité contextuelle"""
        common_keys = set(memory_context.keys()) & set(cue_context.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if memory_context[key] == cue_context[key]:
                similarities.append(1.0)
            else:
                # Similarité partielle pour les valeurs différentes
                similarities.append(0.3)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_time_relevance(self, memory_time: float, time_range: Tuple[float, float]) -> float:
        """Calcule la pertinence temporelle"""
        start_time, end_time = time_range
        if start_time <= memory_time <= end_time:
            return 1.0
        
        # Décroissance exponentielle en dehors de la plage
        time_diff = min(abs(memory_time - start_time), abs(memory_time - end_time))
        decay_rate = 0.1  # Ajustable
        return np.exp(-decay_rate * time_diff)
    
    def _semantic_similarity(self, memory_content: Any, semantic_cue: str) -> float:
        """Calcule la similarité sémantique"""
        # Implémentation basique - à améliorer avec des embeddings
        if isinstance(memory_content, dict) and "concept" in memory_content:
            memory_text = memory_content["concept"]
        else:
            memory_text = str(memory_content)
        
        cue_text = str(semantic_cue)
        
        # Similarité basée sur les mots communs
        memory_words = set(memory_text.lower().split())
        cue_words = set(cue_text.lower().split())
        
        if not memory_words or not cue_words:
            return 0.0
        
        intersection = memory_words & cue_words
        union = memory_words | cue_words
        
        return len(intersection) / len(union)
    
    def _update_memory_accessibility(self, memory: MemoryTrace):
        """Met à jour l'accessibilité d'une mémoire après accès"""
        # Effet de pratique - l'accessibilité augmente avec les accès
        memory.access_count += 1
        memory.last_accessed = time.time()
        
        # Augmentation de l'accessibilité basée sur la force et la récence
        practice_boost = 0.1 * (1.0 - memory.accessibility)
        recency_boost = 0.05 * (1.0 - memory.accessibility)
        
        memory.accessibility = min(1.0, memory.accessibility + practice_boost + recency_boost)
    
    def _calculate_retrieval_confidence(self, memories: List[MemoryTrace], cues: Dict) -> float:
        """Calcule la confiance dans la récupération"""
        if not memories:
            return 0.0
        
        confidence_factors = []
        
        for memory in memories:
            # Confiance basée sur la force et l'accessibilité
            memory_confidence = (memory.strength + memory.accessibility) / 2
            confidence_factors.append(memory_confidence)
        
        # Confiance moyenne pondérée par la pertinence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_context_match(self, memories: List[MemoryTrace], cues: Dict) -> float:
        """Calcule le match contextuel moyen"""
        if not memories or "context" not in cues:
            return 0.0
        
        context_matches = []
        for memory in memories:
            context_match = self._calculate_context_similarity(memory.context, cues["context"])
            context_matches.append(context_match)
        
        return sum(context_matches) / len(context_matches)
    
    def _calculate_emotional_coherence(self, memories: List[MemoryTrace]) -> float:
        """Calcule la cohérence émotionnelle des mémoires récupérées"""
        if len(memories) < 2:
            return 1.0
        
        valences = [memory.valence for memory in memories]
        variance = np.var(valences)
        
        # Cohérence inversement proportionnelle à la variance
        return 1.0 / (1.0 + variance * 10)
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryTrace]:
        """Récupère une mémoire par son ID"""
        for memory_type_dict in self.long_term_memory.values():
            if memory_id in memory_type_dict:
                return memory_type_dict[memory_id]
        return None
    
    def consolidate_memories(self, consolidation_intensity: float = 1.0):
        """
        Processus de consolidation des mémoires
        Renforce les mémoires importantes et élimine les faibles
        """
        consolidation_start = time.time()
        consolidated_count = 0
        forgotten_count = 0
        
        # Consolidation des mémoires actives
        for memory_id in self.consolidation_process["active_consolidation"][:]:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                consolidation_success = self._consolidate_single_memory(memory, consolidation_intensity)
                if consolidation_success:
                    consolidated_count += 1
                    # Retirer de la file si consolidation réussie
                    self.consolidation_process["active_consolidation"].remove(memory_id)
        
        # Processus d'oubli
        for memory_type, memories_dict in self.long_term_memory.items():
            memories_to_remove = []
            
            for memory_id, memory in memories_dict.items():
                # Application de la courbe d'oubli d'Ebbinghaus
                forget_probability = self._calculate_forgetting_probability(memory)
                
                if np.random.random() < forget_probability * consolidation_intensity:
                    memories_to_remove.append(memory_id)
                    forgotten_count += 1
                else:
                    # Renforcement des mémoires fréquemment accédées
                    if memory.access_count > 5:
                        memory.strength = min(1.0, memory.strength + 0.01 * consolidation_intensity)
            
            # Suppression des mémoires oubliées
            for memory_id in memories_to_remove:
                self._forget_memory(memory_id, memory_type)
        
        # Mise à jour du timestamp de consolidation
        self.consolidation_process["last_consolidation_time"] = time.time()
        
        print(f"🔄 Consolidation: {consolidated_count} mémoires consolidées, {forgotten_count} oubliées")
        
        return {
            "consolidated": consolidated_count,
            "forgotten": forgotten_count,
            "duration": time.time() - consolidation_start
        }
    
    def _consolidate_single_memory(self, memory: MemoryTrace, intensity: float) -> bool:
        """Consolide une mémoire individuelle"""
        # Facteurs influençant la consolidation
        consolidation_factors = [
            memory.strength * 0.3,
            memory.valence * self.memory_parameters["emotional_enhancement"] * 0.3,
            memory.accessibility * 0.2,
            (memory.access_count / 10) * 0.2  # Effet de pratique
        ]
        
        consolidation_score = sum(consolidation_factors) * intensity
        
        if consolidation_score > 0.7:
            # Consolidation réussie
            memory.consolidation_state = MemoryConsolidationState.STABLE
            memory.strength = min(1.0, memory.strength + 0.1 * intensity)
            return True
        elif consolidation_score > 0.4:
            # En cours de consolidation
            memory.consolidation_state = MemoryConsolidationState.CONSOLIDATING
            memory.strength = min(1.0, memory.strength + 0.05 * intensity)
            return False
        else:
            # Échec de consolidation
            return False
    
    def _calculate_forgetting_probability(self, memory: MemoryTrace) -> float:
        """Calcule la probabilité d'oubli d'une mémoire"""
        base_forgetting_rate = self.memory_parameters["forgetting_rate"]
        
        # Facteurs réduisant l'oubli
        retention_factors = [
            memory.strength * 0.4,
            abs(memory.valence) * 0.3,  # Mémoires émotionnelles mieux retenues
            (memory.access_count / 20) * 0.2,  # Effet de pratique
            (1.0 if memory.consolidation_state == MemoryConsolidationState.STABLE else 0.5) * 0.1
        ]
        
        retention_score = sum(retention_factors)
        forgetting_prob = base_forgetting_rate * (1.0 - retention_score)
        
        return max(0.0, forgetting_prob)
    
    def _forget_memory(self, memory_id: str, memory_type: MemoryType):
        """Oublie une mémoire spécifique"""
        if memory_id in self.long_term_memory[memory_type]:
            # Suppression des index
            self._remove_from_indexes(memory_id)
            
            # Suppression de la mémoire
            del self.long_term_memory[memory_type][memory_id]
            
            # Mise à jour des métadonnées
            self.memory_metadata["total_memories"] -= 1
            
            print(f"🗑️ Mémoire oubliée: {memory_id}")
    
    def _remove_from_indexes(self, memory_id: str):
        """Supprime une mémoire de tous les index"""
        # Index temporel
        for time_key, memories in self.retrieval_indexes["temporal"].items():
            if memory_id in memories:
                memories.remove(memory_id)
        
        # Index contextuel
        for context_key, memories in self.retrieval_indexes["contextual"].items():
            if memory_id in memories:
                memories.remove(memory_id)
        
        # Index émotionnel
        for emotion_key, memories in self.retrieval_indexes["emotional"].items():
            if memory_id in memories:
                memories.remove(memory_id)
    
    def form_autobiographical_narrative(self) -> Dict[str, Any]:
        """
        Forme un récit autobiographique à partir des mémoires épisodiques
        """
        episodic_memories = list(self.long_term_memory[MemoryType.EPISODIC].values())
        
        if not episodic_memories:
            return {"narrative": "Aucune expérience mémorable encore.", "coherence": 0.0}
        
        # Tri chronologique
        episodic_memories.sort(key=lambda x: x.timestamp)
        
        # Extraction des événements significatifs
        significant_events = [
            mem for mem in episodic_memories 
            if mem.strength > 0.7 or abs(mem.valence) > 0.6
        ]
        
        # Construction du récit
        narrative_parts = []
        total_coherence = 0.0
        
        for i, event in enumerate(significant_events):
            event_description = self._describe_memory_event(event)
            narrative_parts.append(event_description)
            
            # Calcul de la cohérence avec l'événement précédent
            if i > 0:
                prev_event = significant_events[i-1]
                coherence = self._calculate_temporal_coherence(prev_event, event)
                total_coherence += coherence
        
        average_coherence = total_coherence / (len(significant_events) - 1) if len(significant_events) > 1 else 1.0
        
        narrative = " • ".join(narrative_parts)
        
        return {
            "narrative": narrative,
            "coherence": average_coherence,
            "significant_events": len(significant_events),
            "timespan": episodic_memories[-1].timestamp - episodic_memories[0].timestamp
        }
    
    def _describe_memory_event(self, memory: MemoryTrace) -> str:
        """Génère une description textuelle d'un événement mémoire"""
        content_str = str(memory.content)
        
        # Simplification pour l'exemple
        if len(content_str) > 50:
            content_str = content_str[:47] + "..."
        
        emotion_desc = "neutre"
        if memory.valence < -0.3:
            emotion_desc = "négatif"
        elif memory.valence > 0.3:
            emotion_desc = "positif"
        
        return f"[{emotion_desc}] {content_str}"
    
    def _calculate_temporal_coherence(self, event1: MemoryTrace, event2: MemoryTrace) -> float:
        """Calcule la cohérence temporelle entre deux événements"""
        time_gap = event2.timestamp - event1.timestamp
        
        # Cohérence plus élevée pour des événements rapprochés
        if time_gap < 3600:  # 1 heure
            return 0.9
        elif time_gap < 86400:  # 1 jour
            return 0.7
        elif time_gap < 604800:  # 1 semaine
            return 0.5
        else:
            return 0.3
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de mémoire"""
        stats = {
            "total_memories": self.memory_metadata["total_memories"],
            "memory_by_type": {},
            "average_strength": 0.0,
            "average_accessibility": 0.0,
            "consolidation_queue": len(self.consolidation_process["active_consolidation"]),
            "working_memory_load": self._calculate_working_memory_load()
        }
        
        total_strength = 0
        total_accessibility = 0
        memory_count = 0
        
        for memory_type, memories_dict in self.long_term_memory.items():
            type_count = len(memories_dict)
            stats["memory_by_type"][memory_type.value] = type_count
            
            for memory in memories_dict.values():
                total_strength += memory.strength
                total_accessibility += memory.accessibility
                memory_count += 1
        
        if memory_count > 0:
            stats["average_strength"] = total_strength / memory_count
            stats["average_accessibility"] = total_accessibility / memory_count
        
        return stats
    
    def _calculate_working_memory_load(self) -> float:
        """Calcule la charge actuelle de la mémoire de travail"""
        total_items = 0
        total_capacity = 0
        
        for component_name, component in self.working_memory.items():
            if component_name != "central_executive":
                total_items += len(component["contents"])
                total_capacity += component["capacity"]
        
        if total_capacity == 0:
            return 0.0
        
        return total_items / total_capacity

# Test du système de mémoire
if __name__ == "__main__":
    print("💾 TEST DU SYSTÈME DE MÉMOIRE")
    print("=" * 50)
    
    # Création du système
    memory_system = MemorySystem()
    
    # Test d'encodage de mémoires
    test_memories = [
        {
            "content": "Première découverte de la gravité en voyant un objet tomber",
            "type": MemoryType.EPISODIC,
            "context": {"location": "laboratoire", "activity": "observation"},
            "valence": 0.8
        },
        {
            "content": {"concept": "gravité", "definition": "Force d'attraction entre les masses"},
            "type": MemoryType.SEMANTIC, 
            "context": {"domain": "physique", "certainty": "high"},
            "valence": 0.3
        },
        {
            "content": "Procédure pour résoudre des équations simples",
            "type": MemoryType.PROCEDURAL,
            "context": {"skill_level": "beginner", "domain": "mathématiques"},
            "valence": 0.6
        }
    ]
    
    print("\n📝 Encodage des mémoires de test...")
    memory_ids = []
    for mem_data in test_memories:
        mem_id = memory_system.encode_memory(
            content=mem_data["content"],
            memory_type=mem_data["type"],
            context=mem_data["context"],
            valence=mem_data["valence"]
        )
        memory_ids.append(mem_id)
        print(f"Encodé: {mem_id}")
    
    # Test de récupération
    print("\n🔍 Test de récupération...")
    retrieval_result = memory_system.retrieve_memories(
        cues={"context": {"activity": "observation"}},
        memory_type=MemoryType.EPISODIC
    )
    
    print(f"Mémoires récupérées: {len(retrieval_result.memory_traces)}")
    print(f"Confiance: {retrieval_result.confidence:.2f}")
    for memory in retrieval_result.memory_traces:
        print(f" - {memory.content}")
    
    # Test de consolidation
    print("\n🔄 Test de consolidation...")
    consolidation_result = memory_system.consolidate_memories()
    print(f"Résultat: {consolidation_result}")
    
    # Statistiques
    print("\n📊 Statistiques du système:")
    stats = memory_system.get_memory_stats()
    for key, value in stats.items():
        print(f" - {key}: {value}")
    
    # Récit autobiographique
    print("\n📖 Récit autobiographique:")
    narrative = memory_system.form_autobiographical_narrative()
    print(f"Narrative: {narrative['narrative']}")
    print(f"Cohérence: {narrative['coherence']:.2f}")