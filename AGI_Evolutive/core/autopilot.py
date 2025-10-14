
"""High level automation loop for the cognitive architecture."""

from __future__ import annotations

import os
from typing import Optional

from .document_ingest import DocumentIngest
from .persistence import PersistenceManager
from .question_manager import QuestionManager

try:  # Orchestrator is optional when running minimal setups.
    from orchestrator import Orchestrator  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    Orchestrator = None  # type: ignore


class Autopilot:
    """Coordinates ingestion, cognition cycles and persistence."""

    def __init__(
        self,
        arch,
        project_root: Optional[str] = None,
        orchestrator: Optional["Orchestrator"] = None,
    ) -> None:
        self.arch = arch
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.inbox_dir = os.path.join(self.project_root, "inbox")
        self.ingest = DocumentIngest(arch, self.inbox_dir)
        self.persist = PersistenceManager(arch)
        self.questions = QuestionManager(arch)

        if orchestrator is not None:
            self.orchestrator = orchestrator
        elif Orchestrator is not None:
            try:
                self.orchestrator = Orchestrator(arch)
            except Exception:
                self.orchestrator = None
        else:
            self.orchestrator = None

        # Load any previous state so the agent can resume where it left off.
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
        """Return pending auto-generated questions."""

        try:
            return self.questions.pop_questions()
        except Exception:
            return []

    def save_now(self):
        """Force a persistence checkpoint."""

        try:
            return self.persist.save()
        except Exception:
            return False
