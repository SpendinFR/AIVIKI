import time, threading
from typing import Optional

class ReflectionLoop:
    """
    Boucle réflexive périodique (mini "inner monologue").
    """
    def __init__(self, meta_cog, interval_sec: int = 300):
        self.meta = meta_cog
        self.interval = max(30, int(interval_sec))
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running: return
        self.running = True
        def loop():
            while self.running:
                try:
                    a = self.meta.assess_understanding()
                    gaps = []
                    for d in a["domains"].values():
                        gaps.extend(d.get("gaps", []))
                    self.meta.log_inner_monologue(
                        f"Auto-bilan: incertitude={a['uncertainty']:.2f}, gaps={gaps[:3]}",
                        tags=["autonomy","metacognition"]
                    )
                    self.meta.propose_learning_goals(max_goals=2)
                except Exception as e:
                    self.meta.log_inner_monologue(f"Reflection loop error: {e}", tags=["error"])
                time.sleep(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
