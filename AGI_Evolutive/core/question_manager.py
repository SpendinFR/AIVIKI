
# core/question_manager.py
"""
QuestionManager: génère des questions à poser à l'utilisateur
selon l'incertitude, les lacunes mémorielles et la curiosité.
- Local: expose une file 'pending_questions' que la CLI peut afficher.
- Politique simple: si incertitude > seuil ou info manquante -> poser 1-2 questions ciblées.
"""
import time
from typing import List, Dict, Any

class QuestionManager:
    def __init__(self, arch):
        self.arch = arch
        self.pending_questions: List[Dict[str, Any]] = []
        self.max_queue = 10
        self.last_generated = 0.0
        self.cooldown = 10.0  # secondes minimum entre générations
    
    def _metacognitive_uncertainty(self) -> float:
        meta = getattr(self.arch, "metacognition", None)
        try:
            return 1.0 - meta.metacognitive_states.get("awareness_level", 0.5)
        except Exception:
            return 0.5
    
    def _goal_information_need(self) -> float:
        goals = getattr(self.arch, "goals", None)
        try:
            return max(0.0, 1.0 - goals.goal_history.get("goal_achievement_rate", 0.0))
        except Exception:
            return 0.3
    
    def maybe_generate_questions(self):
        now = time.time()
        if now - self.last_generated < self.cooldown:
            return
        need = max(self._metacognitive_uncertainty(), self._goal_information_need())
        if need < 0.4:
            return
        qs = []
        if need >= 0.7:
            qs.append({"type": "clarify_goal", "text": "Quel est l’objectif prioritaire sur lequel tu veux que je me concentre maintenant ?"})
            qs.append({"type": "request_docs", "text": "As-tu un document ou un exemple pour m’aider à mieux comprendre le contexte ?"})
        else:
            qs.append({"type": "missing_info", "text": "Peux-tu préciser le contexte ou les contraintes principales liées à ta demande ?"})
        for q in qs:
            if len(self.pending_questions) >= self.max_queue:
                self.pending_questions.pop(0)
            self.pending_questions.append(q)
        self.last_generated = now
    
    def pop_questions(self) -> List[Dict[str, Any]]:
        out = list(self.pending_questions)
        self.pending_questions.clear()
        return out
