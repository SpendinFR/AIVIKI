from __future__ import annotations

import threading
import time
from typing import Optional


class AutonomyCore:
    """Minimal autonomous loop placeholder."""

    def __init__(self, arch, logger, goal_dag):
        self.arch = arch
        self.logger = logger
        self.goal_dag = goal_dag
        self._idle_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def notify_user_activity(self) -> None:
        self._idle_event.set()

    def _loop(self) -> None:
        while self._running:
            if self._idle_event.wait(timeout=60):
                self._idle_event.clear()
                continue
            try:
                goal = self.goal_dag.choose_next_goal()
                if self.logger:
                    self.logger.write("autonomy.idle", goal=goal)
            except Exception:
                pass
            time.sleep(5)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
