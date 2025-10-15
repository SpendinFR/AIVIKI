from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os, json, time, uuid, math, re
from collections import deque
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

    # --- Extensions multi-domaines ---
    def domain_reports(self) -> Dict[str, Dict[str, Any]]:
        domains = sorted({row.get("domain", "global") for row in self._iter() if "success" in row})
        return {d: self.report(d if d != "global" else None) for d in domains}

    def dynamic_threshold(self, domain: Optional[str] = None, base: float = 0.45) -> float:
        info = self.report(domain)
        count = info.get("count", 0) or 0
        ece = info.get("ece")
        threshold = base
        if ece is not None:
            threshold += max(-0.15, min(0.20, ece * 0.6))
        if count > 80:
            threshold -= 0.05
        elif count < 15:
            threshold += 0.05
        return max(0.1, min(0.85, threshold))

    def should_abstain(self, domain: Optional[str], confidence: float, margin: float = 0.05) -> bool:
        threshold = self.dynamic_threshold(domain)
        return float(confidence) + margin < threshold


class NoveltyDetector:
    """Very light-weight detector for out-of-distribution user inputs."""

    def __init__(self, window: int = 128, threshold: float = 0.45) -> None:
        self.window = deque(maxlen=max(8, window))
        self.threshold = max(0.05, min(0.95, threshold))

    # Feature extraction intentionally simple (token stats + punctuation balance)
    def _features(self, text: str) -> Tuple[float, ...]:
        text = text or ""
        tokens = re.findall(r"\w+", text.lower())
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / max(1, token_count)
        char_count = len(text)
        digit_ratio = sum(c.isdigit() for c in text) / max(1, char_count)
        upper_ratio = sum(c.isupper() for c in text) / max(1, char_count)
        punctuation_ratio = sum(c in "?!;:" for c in text) / max(1, char_count)
        long_token_ratio = sum(len(t) >= 10 for t in tokens) / max(1, token_count)
        question_tokens = sum(1 for t in tokens if t in {"pourquoi", "comment", "qui", "quoi", "où", "how", "why"})
        question_density = question_tokens / max(1, token_count)
        length_norm = min(1.0, token_count / 160.0)
        return (
            length_norm,
            unique_ratio,
            digit_ratio,
            upper_ratio,
            punctuation_ratio,
            long_token_ratio,
            question_density,
        )

    def _avg_vector(self) -> Optional[Tuple[float, ...]]:
        if not self.window:
            return None
        dims = len(self.window[0])
        sums = [0.0] * dims
        for vec in self.window:
            for i, value in enumerate(vec):
                sums[i] += value
        return tuple(s / len(self.window) for s in sums)

    def novelty_score(self, text: str) -> float:
        vec = self._features(text)
        avg = self._avg_vector()
        if avg is None:
            return 0.0
        dist = math.sqrt(sum((vec[i] - avg[i]) ** 2 for i in range(len(vec))))
        return max(0.0, min(1.0, dist * 2.2))

    def assess(self, text: str, update: bool = True) -> Tuple[float, bool]:
        score = self.novelty_score(text)
        flagged = score >= self.threshold
        if update:
            self.window.append(self._features(text))
        return score, flagged
