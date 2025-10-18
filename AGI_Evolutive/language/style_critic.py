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
