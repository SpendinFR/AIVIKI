import time, threading
from typing import Any, Dict, List, Optional

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

    def test_hypotheses(self, scratch: Dict[str, Any], max_tests: int = 3) -> Dict[str, Any]:
        """Génère quelques hypothèses, tente un contre-exemple pour chacune."""

        scratch = scratch or {}
        try:
            max_tests = max(1, int(max_tests))
        except Exception:
            max_tests = 3

        observation = (
            scratch.get("observation")
            or scratch.get("focus")
            or scratch.get("text")
            or scratch.get("question")
        )

        memory = getattr(self.meta, "memory", None)
        recent: List[Dict[str, Any]] = []
        if memory and hasattr(memory, "get_recent_memories"):
            try:
                recent = memory.get_recent_memories(n=80) or []
            except Exception:
                recent = []
        if not observation and recent:
            last = recent[-1]
            observation = last.get("text") or last.get("content")

        arch = scratch.get("architecture") or scratch.get("arch")
        if arch is None:
            arch = getattr(self.meta, "architecture", None)
        if arch is None:
            arch = getattr(self.meta, "arch", None)

        abduction = scratch.get("abduction")
        if abduction is None and arch is not None:
            abduction = getattr(arch, "abduction", None)

        hypotheses: List[Dict[str, Any]] = []
        if abduction and observation:
            try:
                generated = abduction.generate(observation) or []
                for hyp in generated[:max_tests]:
                    label = getattr(hyp, "label", None) or getattr(hyp, "name", None) or str(hyp)
                    explanation = getattr(hyp, "explanation", "")
                    score = getattr(hyp, "score", 0.0)
                    ask_next = getattr(hyp, "ask_next", None)

                    counterexample = None
                    label_lower = label.lower() if isinstance(label, str) else ""
                    for memo in reversed(recent):
                        text = str(memo.get("text") or memo.get("content") or "").lower()
                        if not text or (label_lower and label_lower not in text):
                            continue
                        if any(token in text for token in ("pas", "non", "jamais", "faux", "wrong", "erreur")):
                            counterexample = {
                                "id": memo.get("id") or memo.get("memory_id"),
                                "text": memo.get("text") or memo.get("content"),
                                "ts": memo.get("ts") or memo.get("t"),
                            }
                            break

                    hypotheses.append(
                        {
                            "label": label,
                            "score": float(score) if isinstance(score, (int, float)) else 0.0,
                            "explanation": explanation,
                            "ask_next": ask_next,
                            "counterexample": counterexample,
                        }
                    )
            except Exception:
                hypotheses = []

        if not hypotheses:
            fallback_label = observation or scratch.get("reason") or "hypothèse_manquante"
            hypotheses = [
                {
                    "label": str(fallback_label),
                    "score": 0.4,
                    "explanation": "Fallback généré faute d'abduction.",
                    "ask_next": None,
                    "counterexample": None,
                }
            ]

        hypotheses = hypotheses[:max_tests]
        contradicted = sum(1 for h in hypotheses if h.get("counterexample"))
        summary = (
            f"{len(hypotheses)} hypothèse(s) testée(s), {contradicted} contre-exemple(s) détecté(s)."
        )

        return {"tested": len(hypotheses), "hypotheses": hypotheses, "summary": summary}
