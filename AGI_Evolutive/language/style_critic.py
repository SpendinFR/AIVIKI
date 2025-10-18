from __future__ import annotations

import re
from typing import Dict, List


class StyleCritic:
    """Critique légère du style pour post-traiter les réponses."""

    def __init__(self, max_chars: int = 1200):
        self.max_chars = int(max_chars)

    def analyze(self, text: str) -> Dict[str, object]:
        sample = (text or "").strip()
        length = len(sample)
        lines: List[str] = sample.splitlines() or [""]
        flags: List[str] = []

        if length > self.max_chars:
            flags.append("trim")
        if any(len(line) > 320 for line in lines):
            flags.append("long_line")
        if "  " in sample:
            flags.append("double_space")

        return {
            "length": length,
            "line_count": len(lines),
            "flags": flags,
        }

    def rewrite(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip()
            if not cleaned.endswith("."):
                cleaned = cleaned.rstrip("…") + "…"
        return cleaned
import re
from typing import Dict, Any


class StyleCritic:
    """
    Checklist rapide + petites réécritures locales:
    - Couper sur-longues
    - Adoucir/assumer modaux
    - Dé-doublonner adverbes
    - Éviter répétitions proches
    """

    def __init__(self, max_chars=1200):
        self.max_chars = max_chars

    def analyze(self, text: str) -> Dict[str, Any]:
        issues = []
        if len(text) > self.max_chars:
            issues.append(("too_long", len(text)))
        if text.count("!!") > 0:
            issues.append(("excess_bang", text.count("!!")))
        if re.search(r"\b(très\s+){2,}", text, flags=re.I):
            issues.append(("adverb_dup", 1))
        if re.search(r"\b(peut[-\s]?être|peut etre)\b", text, flags=re.I):
            issues.append(("hedging_maybe", 1))
        return {"issues": issues}

    def rewrite(self, text: str) -> str:
        t = text
        # coupe douce si trop long
        if len(t) > self.max_chars:
            t = t[: self.max_chars].rstrip()
            if not t.endswith((".", "!", "?")):
                t += "…"
        # modaux
        t = re.sub(r"\b(peut[-\s]?être|peut etre)\b", "probablement", t, flags=re.I)
        # adverbes en double
        t = re.sub(r"\b(très)\s+(très)\b", r"\1", t, flags=re.I)
        # ponctuation
        t = re.sub(r"\s+([!?;:])", r"\1", t)  # pas d'espace avant
        t = re.sub(r"([!?;:])(?=\S)", r"\1 ", t)  # espace après
        return t
