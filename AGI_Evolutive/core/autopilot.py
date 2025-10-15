
"""High level automation loop for the cognitive architecture."""

from __future__ import annotations

import os
from typing import Any, Optional

from .document_ingest import DocumentIngest
from .persistence import PersistenceManager
from .question_manager import QuestionManager


class Autopilot:
    """Coordinates ingestion, cognition cycles and persistence."""

    def __init__(
        self,
        arch,
        project_root: Optional[str] = None,
        orchestrator: Optional[Any] = None,
    ) -> None:
        self.arch = arch
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.inbox_dir = os.path.join(self.project_root, "inbox")
        self.ingest = DocumentIngest(arch, self.inbox_dir)
        self.persist = PersistenceManager(arch)
        self.questions = QuestionManager(arch)

        self.orchestrator = orchestrator

        try:
            self.persist.load()
        except Exception:
            # Persistence is a best-effort facility; failures should not crash boot.
            pass

    def step(self, user_msg: Optional[str] = None):
        """Run a full autopilot iteration."""

        # 1) Integrate any freshly dropped documents.
        try:
            self.ingest.integrate()
        except Exception:
            pass

        # 2) Execute one cognitive cycle.
        out = None
        try:
            out = self.arch.cycle(user_msg=user_msg, inbox_docs=None)
        except Exception:
            out = None

        # 3) Allow the optional orchestrator to do additional coordination.
        if self.orchestrator is not None:
            try:
                self.orchestrator.run_once_cycle(user_msg=user_msg)
            except Exception:
                # L'orchestrateur est auxiliaire : une erreur ne doit pas
                # interrompre la boucle principale.
                pass

        # 4) Maybe create follow-up questions for the user.
        try:
            self.questions.maybe_generate_questions()
        except Exception:
            pass

        # 5) Persist the current state regularly.
        try:
            self.persist.autosave_tick()
        except Exception:
            pass

        return out

    def pending_questions(self):
        """Retourne les questions auto-générées, + celles de validation d'apprentissage."""
        # Celles déjà générées par policies/métacog
        try:
            qs = self.questions.pop_questions()
        except Exception:
            qs = []

        # Merge avec les demandes de validation stockées en mémoire
        try:
            mem = getattr(self.arch, 'memory', None)
            seen = set(q.get('text','') for q in qs)
            if mem and hasattr(mem, 'get_recent_memories'):
                recents = mem.get_recent_memories(50)
                for item in recents:
                    if item.get('kind') == 'validation_request':
                        text = item.get('content') or item.get('text') or 'Valider un apprentissage'
                        if text not in seen:
                            qs.append({"type": "validation", "text": text})
                            seen.add(text)
                        if len(qs) >= 5:
                            break
                if len(qs) < 5:
                    for item in recents:
                        if item.get('kind') == 'question_active':
                            text = item.get('content') or item.get('text') or ''
                            if not text or text in seen:
                                continue
                            qs.append({"type": "active", "text": text})
                            seen.add(text)
                            if len(qs) >= 5:
                                break
        except Exception:
            pass
        return qs

    def save_now(self):
        """Force a persistence checkpoint."""

        try:
            return self.persist.save()
        except Exception:
            return False
