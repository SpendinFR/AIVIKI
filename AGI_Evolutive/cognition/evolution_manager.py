import os, json, time, statistics
from typing import Dict, Any, List

class EvolutionManager:
    """
    Suit les performances de cycle en cycle et propose des ajustements macro.
    Persiste dans data/evolution.json
    """
    def __init__(self, data_path: str = "data/evolution.json"):
        self.path = data_path
        self.state = {
            "cycle_count": 0,
            "metrics_history": [],   # [{"ts":..., "intr":..,"extr":..,"learn":..,"uncert":..}]
            "strategies": []         # notes d'ajustement
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def log_cycle(self, intrinsic: float, extrinsic: float, learning_rate: float, uncertainty: float):
        self.state["cycle_count"] += 1
        self.state["metrics_history"].append({
            "ts": time.time(),
            "intr": float(intrinsic),
            "extr": float(extrinsic),
            "learn": float(learning_rate),
            "uncert": float(uncertainty)
        })
        self.state["metrics_history"] = self.state["metrics_history"][-500:]
        self._save()

    def propose_macro_adjustments(self) -> List[str]:
        mh = self.state["metrics_history"]
        if len(mh) < 10: return []
        last = mh[-10:]
        avg_unc = statistics.fmean(x["uncert"] for x in last)
        avg_learn = statistics.fmean(x["learn"] for x in last)
        notes: List[str] = []
        if avg_unc > 0.65:
            notes.append("Augmenter exploration (curiosity), planifier plus de questions ciblées.")
        if avg_learn < 0.45:
            notes.append("Changer stratégie d’étude: plus d’exemples concrets et feedback.")
        if not notes:
            notes.append("Maintenir les stratégies actuelles, progression stable.")
        self.state["strategies"].append({"ts": time.time(), "notes": notes})
        self._save()
        return notes
