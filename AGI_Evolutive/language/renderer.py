from __future__ import annotations
from typing import Dict, Any
import random

class LanguageRenderer:
    def __init__(self, voice_profile, lexicon):
        self.voice = voice_profile
        self.lex = lexicon

    def _decorate_with_voice(self, text: str) -> str:
        st = self.voice.style()
        # exemples simples (tu peux enrichir)
        if st["emoji"] > 0.5:
            text = "🙂 " + text
        if st["formality"] > 0.7:
            text = "Bonjour, " + text
        if st["warmth"] > 0.7 and not text.endswith("!"):
            text += " (je suis là si besoin)"
        return text

    def _attach_context_snippets(self, text: str, ctx: Dict[str, Any]) -> str:
        # Ajouter « tu m’as dit le 12/09… »
        moments = ctx.get("key_moments") or []
        if moments:
            pick = random.choice(moments[-3:])
            text += f"\n\n↪ En lien: {pick}"
        return text

    def _lexical_variation(self, text: str, novelty: float = 0.3) -> str:
        # insère une collocation “aimée” de temps en temps
        phrase = self.lex.sample_collocation(novelty=novelty)
        if phrase:
            text += f"\n\n({phrase})"
        return text

    def render_reply(self, semantics: Dict[str, Any], ctx: Dict[str, Any]) -> str:
        base = (semantics.get("text") or "").strip()
        if not base:
            base = "Je te réponds en tenant compte de notre historique."

        # conservatisme vs nouveauté selon voix
        novelty = 0.2 + 0.5 * self.voice.style().get("storytelling", 0.0)

        out = self._decorate_with_voice(base)
        out = self._attach_context_snippets(out, ctx)
        out = self._lexical_variation(out, novelty=novelty)
        return out
