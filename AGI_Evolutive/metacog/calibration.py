from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, json, time, uuid
from statistics import mean

class CalibrationMeter:
    def __init__(self, path: str = "data/calibration.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def log_prediction(self, domain: str, p: float, meta: Optional[Dict[str, Any]] = None) -> str:
        eid = str(uuid.uuid4())
        row = {"id": eid, "t": time.time(), "domain": domain, "p": float(p), "meta": meta or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return eid

    def log_outcome(self, event_id: str, success: bool) -> None:
        items = []
        if os.path.exists(self.path):
            items = [json.loads(l) for l in open(self.path, "r", encoding="utf-8")]
        for it in items:
            if it.get("id") == event_id:
                it["success"] = bool(success); break
        with open(self.path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def _iter(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path): return []
        return [json.loads(l) for l in open(self.path, "r", encoding="utf-8")]

    def report(self, domain: Optional[str] = None) -> Dict[str, Any]:
        rows = [r for r in self._iter() if "success" in r and (domain is None or r.get("domain")==domain)]
        if not rows:
            return {"count": 0, "brier": None, "ece": None, "bins": []}
        brier = mean([(float(r["p"]) - (1.0 if r["success"] else 0.0))**2 for r in rows])
        # ECE: 10 bins
        bins = [[] for _ in range(10)]
        for r in rows:
            i = min(9, max(0, int(float(r["p"])*10)))
            bins[i].append(r)
        ece_terms = []
        for i, b in enumerate(bins):
            if not b: continue
            conf = mean([float(x["p"]) for x in b])
            acc = mean([1.0 if x["success"] else 0.0 for x in b])
            ece_terms.append(abs(conf - acc) * len(b)/len(rows))
        ece = sum(ece_terms) if ece_terms else None
        return {"count": len(rows), "brier": brier, "ece": ece, "bins": [len(b) for b in bins]}

    def suggested_hedging_delta(self, domain: Optional[str] = None) -> float:
        r = self.report(domain)
        if r["ece"] is None: return 0.0
        if r["ece"] > 0.25:  return +0.15
        if r["ece"] > 0.15:  return +0.08
        if r["ece"] < 0.05:  return -0.04
        return 0.0
