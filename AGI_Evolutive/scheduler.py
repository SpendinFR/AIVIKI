import time
from typing import Callable, Dict, Any

class Scheduler:
    """
    Planificateur léger: tick-driven (appel périodique), sans threads obligatoires.
    register_job(name, interval_sec, func); puis scheduler.tick() dans ta boucle.
    """
    def __init__(self):
        self.jobs: Dict[str, Dict[str,Any]] = {}

    def register_job(self, name: str, interval_sec: int, func: Callable[[], None]):
        self.jobs[name] = {"interval": max(5, int(interval_sec)), "func": func, "last": 0.0}

    def tick(self):
        now = time.time()
        for name, job in self.jobs.items():
            if now - job["last"] >= job["interval"]:
                try:
                    job["func"]()
                finally:
                    job["last"] = now
