
# core/autopilot.py
"""
Autopilot: orchestre un pas de boucle cognitive
- intègre documents (inbox)
- déclenche un cycle de l'architecture
- génère des questions si besoin
- autosave via PersistenceManager
"""
import os
from typing import Optional
from .persistence import PersistenceManager
from .document_ingest import DocumentIngest
from .question_manager import QuestionManager
from orchestrator import Orchestrator

class Autopilot:
    def __init__(self, arch, project_root: Optional[str] = None, orchestrator: Optional[Orchestrator] = None):
        self.arch = arch
        self.project_root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.inbox_dir = os.path.join(self.project_root, "inbox")
        self.ingest = DocumentIngest(arch, self.inbox_dir)
        self.persist = PersistenceManager(arch)
        self.questions = QuestionManager(arch)
        self.orchestrator = orchestrator or Orchestrator(arch)
        # charger un état si disponible
        self.persist.load()

    def step(self, user_msg: Optional[str] = None):
        # 1) intégrer docs nouveaux
        self.ingest.integrate()
        # 2) appel d'un cycle cognitif
        out = self.arch.cycle(user_msg=user_msg, inbox_docs=None)
        # 2b) orchestrateur étendu
        if self.orchestrator is not None:
            self.orchestrator.run_once_cycle(user_msg=user_msg)
        # 3) générer éventuellement des questions
        self.questions.maybe_generate_questions()
        # 4) autosave
        self.persist.autosave_tick()
        return out
    
    def pending_questions(self):
        return self.questions.pop_questions()
    
    def save_now(self):
        return self.persist.save()
