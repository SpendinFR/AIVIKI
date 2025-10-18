# 🚀 main.py - Point d'entrée AGI Évolutive
import glob
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

try:
    from AGI_Evolutive.language.quote_memory import QuoteMemory  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    QuoteMemory = None  # type: ignore

try:
    from AGI_Evolutive.language.social.tactic_selector import TacticSelector  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    TacticSelector = None  # type: ignore

try:
    from AGI_Evolutive.language.ranker import RankerModel  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    RankerModel = None  # type: ignore

try:
    from AGI_Evolutive.language.inbox_ingest import ingest_inbox_paths  # type: ignore
except ImportError:  # pragma: no cover - module optionnel
    ingest_inbox_paths = None  # type: ignore

# --- Questions CLI helpers ---
def _get_qm(auto) -> Any:
    # Essaie plusieurs emplacements possibles
    return (
        getattr(auto, "question_manager", None)
        or getattr(auto, "questions", None)
        or getattr(getattr(auto, "arch", None), "question_manager", None)
        or getattr(getattr(auto, "arch", None), "questions", None)
    )


def _print_pending(
    qm, k: int = 3, preset: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Affiche les k dernières questions, renvoie la même liste (ordre d'affichage)."""
    if preset is not None:
        pending = list(preset)
    elif not qm:
        return []
    else:
        pending = list(getattr(qm, "pending_questions", []))
    if not pending:
        return []
    # on prend les k dernières (les plus récentes) et on garde l'ordre d’affichage
    view = pending[-k:]
    print("\n— Questions en attente —")
    for i, q in enumerate(view, 1):
        qtype = q.get("type", "?")
        score = q.get("score", 0.0)
        text = q.get("text", "")
        print(f"[{i}] ({qtype}, score={score:.2f}) {text}")
    print("Réponds avec : a <num> <ta réponse>   ex:  a 2 oui, c’était volontaire\n")
    return view



from AGI_Evolutive.core.autopilot import Autopilot
from AGI_Evolutive.core.cognitive_architecture import CognitiveArchitecture
from AGI_Evolutive.cognition.prioritizer import GoalPrioritizer
from AGI_Evolutive.orchestrator import Orchestrator
from AGI_Evolutive.language.voice import VoiceProfile
from AGI_Evolutive.language.lexicon import LiveLexicon
from AGI_Evolutive.language.style_observer import StyleObserver
from AGI_Evolutive.conversation.context import ContextBuilder
from AGI_Evolutive.language.renderer import LanguageRenderer
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.preferences_adapter import PreferencesAdapter

BANNER = """
╔══════════════════════════════════════════════╗
║            🧠  AGI ÉVOLUTIVE v1.0            ║
║  Architecture cognitive intégrée & évolutive ║
╚══════════════════════════════════════════════╝
"""
HELP_TEXT = """
Commandes disponibles :
  /help        → afficher cette aide
  /inbox       → liste les fichiers déposés dans ./inbox
  /save        → force une sauvegarde immédiate
  /state       → montre les infos d'état globales
  /quit        → quitte proprement
Astuce : déposez vos fichiers (.txt, .md, .json, etc.) dans ./inbox/
         ils seront intégrés automatiquement en mémoire.
"""

def list_inbox(inbox_dir="inbox"):
    files = [os.path.basename(p) for p in glob.glob(os.path.join(inbox_dir, "*"))]
    if not files:
        print("📂 Inbox vide.")
    else:
        print("📁 Inbox :", ", ".join(files))

def run_cli():
    print(BANNER)
    print("Chargement de l'architecture cognitive…")
    try:
        arch = CognitiveArchitecture()
        arch.prioritizer = getattr(arch, "prioritizer", GoalPrioritizer(arch))
        # --- bootstrap voix & contexte ---
        arch.voice_profile = getattr(
            arch,
            "voice_profile",
            VoiceProfile(arch.self_model, user_model=getattr(arch, "user_model", None)),
        )
        arch.lexicon = getattr(arch, "lexicon", LiveLexicon())
        arch.style_observer = getattr(
            arch,
            "style_observer",
            StyleObserver(
                arch.self_model,
                getattr(arch, "homeostasis", None),
                arch.voice_profile,
                arch.lexicon,
                user_model=getattr(arch, "user_model", None),
            ),
        )
        # Harmonise les instances utilisées par le renderer et l'observateur de style
        if getattr(arch.style_observer, "voice", None) is not None:
            arch.voice_profile = arch.style_observer.voice
        if getattr(arch.style_observer, "lex", None) is not None:
            arch.lexicon = arch.style_observer.lex
        arch.context_builder = getattr(arch, "context_builder", ContextBuilder(arch))
        arch.renderer = getattr(
            arch,
            "renderer",
            LanguageRenderer(arch.voice_profile, arch.lexicon),
        )
        if getattr(arch.renderer, "voice", None) is not arch.voice_profile:
            arch.renderer.voice = arch.voice_profile
        if getattr(arch.renderer, "lex", None) is not arch.lexicon:
            arch.renderer.lex = arch.lexicon
        # --- fin bootstrap ---

        # 10.1 — rattache les modules langue avancés si disponibles
        qm = None
        if QuoteMemory:
            try:
                qm = QuoteMemory()
            except Exception:
                qm = None
        if qm:
            if hasattr(arch, "voice_profile"):
                arch.voice_profile.quote_memory = qm
            else:
                arch.quote_memory = qm

        if TacticSelector and not hasattr(arch, "tactic_selector"):
            try:
                arch.tactic_selector = TacticSelector()
            except Exception:
                arch.tactic_selector = None

        if RankerModel and not hasattr(arch, "ranker"):
            try:
                arch.ranker = RankerModel()
            except Exception:
                arch.ranker = None

        if getattr(arch, "renderer", None) and getattr(arch, "ranker", None):
            try:
                arch.renderer.ranker = arch.ranker
                if hasattr(arch, "voice_profile"):
                    arch.renderer.voice = arch.voice_profile
                if hasattr(arch, "lexicon"):
                    arch.renderer.lex = arch.lexicon
            except Exception:
                pass

        # Optionnel : ingestion ciblée de l'inbox au démarrage
        # if ingest_inbox_paths:
        #     ingest_inbox_paths(["inbox/foo.txt", "inbox/bar.md"], arch=arch)

        orc = Orchestrator(arch)
        auto = Autopilot(arch, orchestrator=orc)
    except Exception as e:
        print("❌ Erreur d'initialisation :", e)
        traceback.print_exc()
        sys.exit(1)

    voice = arch.voice_profile
    concept_extractor = ConceptExtractor(getattr(getattr(arch, "memory", None), "store", None))
    prefs = PreferencesAdapter(getattr(arch, "beliefs_graph", None))
    _last_cleanup_ts = 0.0

    print("✅ AGI initialisée. (Persistance & mémoire prêtes)")
    print(HELP_TEXT)
    print("🗨️  Démarrez la conversation ou tapez /help.")

    _last_view: List[Dict[str, Any]] = []
    _pending_cache: List[Dict[str, Any]] = []

    while True:
        try:
            # Affiche jusqu'à 3 questions en attente à chaque itération
            try:
                qm = _get_qm(auto)
                preset = _pending_cache if _pending_cache else None
                _last_view = _print_pending(qm, k=3, preset=preset)  # garde en mémoire locale
            except Exception:
                _last_view = []

            try:
                now = time.time()
                if now - _last_cleanup_ts > 24 * 3600:
                    mem_store = getattr(getattr(arch, "memory", None), "store", None) or getattr(arch, "memory", None)
                    from AGI_Evolutive.memory.janitor import run_once as memory_janitor_run

                    if mem_store:
                        memory_janitor_run(mem_store)
                    _last_cleanup_ts = now
            except Exception:
                pass

            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⏳ Sauvegarde avant sortie…")
            try:
                auto.save_now()
            except Exception as e:
                print("⚠️ Erreur lors de la sauvegarde :", e)
            print("👋 Fin de session.")
            break

        # --- Réponse à une question : "a <num> <réponse>" / "answer <num> ..."
        m = re.match(r"^\s*(a|answer|reponds?|réponds?)\s+(\d+)\s+(.+)$", msg, flags=re.IGNORECASE)
        if m:
            idx = max(0, int(m.group(2)) - 1)
            answer_text = m.group(3).strip()
            qm = _get_qm(auto)
            if not qm:
                print("⚠️  Aucun gestionnaire de questions accessible.")
                continue

            # Récupère la vue actuelle (ou replie sur les pending)
            view = _last_view if _last_view else list(getattr(qm, "pending_questions", []))[-3:]
            if not view or idx >= len(view):
                print("⚠️  Index hors limites. Tape 'q' pour lister.")
                continue

            q = view[idx]
            qid = q.get("id") or q.get("qid") or q.get("uuid")

            # 1) Ingestion de ta réponse comme utterance utilisateur (traçabilité)
            try:
                # Préfère ton interface de perception si dispo
                per = getattr(getattr(auto, "arch", None), "perception", None)
                meta = {"answer_to": qid, "question_text": q.get("text", ""), "ts": time.time()}
                if per and hasattr(per, "ingest_user_utterance"):
                    per.ingest_user_utterance(answer_text, author="user", meta=meta)
                else:
                    memory = getattr(getattr(auto, "arch", None), "memory", None)
                    if memory and hasattr(memory, "add_memory"):
                        memory.add_memory(
                            {
                                "kind": "user_answer",
                                "q_id": qid,
                                "q_text": q.get("text", ""),
                                "text": answer_text,
                                "ts": time.time(),
                            }
                        )
            except Exception:
                pass

            # 2) Notifie le QuestionManager si une API existe
            updated = False
            for meth in ("record_answer", "resolve_question", "set_answer"):
                if hasattr(qm, meth):
                    try:
                        getattr(qm, meth)(qid, answer_text)
                        updated = True
                        break
                    except Exception:
                        pass

            # 3) Fallback : on retire manuellement la question de la file
            if not updated:
                try:
                    pend = getattr(qm, "pending_questions", [])
                    # enlève la 1re occurrence correspondante
                    for i in range(len(pend) - 1, -1, -1):
                        if (pend[i].get("id") or pend[i].get("qid")) == qid or pend[i] is q:
                            pend.pop(i)
                            break
                except Exception:
                    pass

            try:
                if _pending_cache:
                    _pending_cache = [
                        item
                        for item in _pending_cache
                        if not (
                            (item.get("id") or item.get("qid") or item.get("uuid")) == qid
                            or item is q
                        )
                    ]
            except Exception:
                pass

            print(f"✅  Réponse enregistrée pour [{idx+1}] : {answer_text}")
            # laisse la boucle continuer (l’abduction/planification la prendra au prochain tick)
            continue
        # --- fin: réponse NL ---

        if not msg:
            continue

        # ==== COMMANDES ====
        if msg in ("/quit", "/exit"):
            print("💾 Sauvegarde finale…")
            auto.save_now()
            print("👋 À bientôt.")
            break

        elif msg == "/help":
            print(HELP_TEXT)
            continue

        elif msg == "/inbox":
            list_inbox()
            continue

        elif msg == "/save":
            path = auto.save_now()
            print(f"💾 Snapshot sauvegardé : {path}")
            continue

        elif msg == "/state":
            try:
                t = time.strftime("%H:%M:%S", time.localtime())
                total_mem = getattr(arch.memory, "memory_metadata", {}).get("total_memories", 0)
                print(f"🧩 Heure locale: {t}")
                print(f"🧠 Mémoires stockées : {total_mem}")
                print(f"⚙️  Dernière sauvegarde : {time.strftime('%H:%M:%S', time.localtime(auto.persist._last_save))}")
            except Exception as e:
                print("⚠️ Impossible d'afficher l'état :", e)
            continue

        # ==== INTERACTION ====
        t = msg.lower()
        if any(kw in t for kw in ["++", "parfait", "top", "merci beaucoup", "exactement"]):
            voice.update_from_feedback(msg, positive=True)
        elif any(kw in t for kw in ["trop long", "bof", "pas clair", "trop familier", "trop froid"]):
            voice.update_from_feedback(msg, positive=False)

        try:
            t = msg.lower()
            pos = any(kw in t for kw in ["++", "parfait", "top", "merci beaucoup", "exactement"])
            neg = any(kw in t for kw in ["trop long", "bof", "pas clair", "trop familier", "trop froid"])
            if pos or neg:
                sign = 1 if pos and not neg else -1
                raw_concepts = concept_extractor._extract_concepts(msg) or []
                targets = [str(c).strip().lower() for c in raw_concepts if c and len(str(c)) >= 3][:5]
                if targets:
                    evidence_id = f"user:{int(time.time())}"
                    strength = 1.0 if sign > 0 else 0.8
                    for c in targets:
                        prefs.observe_feedback(concept=c, sign=sign, evidence_id=evidence_id, strength=strength)

                selector = getattr(arch, "tactic_selector", None)
                if selector and hasattr(selector, "feedback"):
                    try:
                        arm = getattr(arch, "_last_macro", None)
                        if pos and not neg:
                            selector.feedback(+1.0, arm=arm)
                        elif neg and not pos:
                            selector.feedback(-1.0, arm=arm)
                    except Exception:
                        pass

                qm_for_feedback = None
                if hasattr(arch, "voice_profile"):
                    qm_for_feedback = getattr(arch.voice_profile, "quote_memory", None)
                if qm_for_feedback is None:
                    qm_for_feedback = getattr(arch, "quote_memory", None)
                if qm_for_feedback:
                    try:
                        if pos and not neg:
                            qm_for_feedback.reward_last(+1.0)
                        if neg and not pos:
                            qm_for_feedback.reward_last(-1.0)
                        qm_for_feedback.save()
                    except Exception:
                        pass

                pack = getattr(arch, "_last_candidates", None)
                if pack and isinstance(pack, dict) and pack.get("alts"):
                    try:
                        ctx_for_rank = {}
                        vp = getattr(arch, "voice_profile", None)
                        style_policy = getattr(vp, "style_policy", None)
                        if style_policy and hasattr(style_policy, "params"):
                            ctx_for_rank["style"] = style_policy.params
                        ranker = getattr(arch, "ranker", None)
                        if ranker and hasattr(ranker, "update_pair"):
                            winner = (pack.get("chosen") or {}).get("text") or pack.get("text")
                            alts = pack.get("alts") or []
                            loser = None
                            if alts:
                                loser = (alts[0] or {}).get("text") if isinstance(alts[0], dict) else None
                            if winner and loser:
                                if pos and not neg:
                                    ranker.update_pair(ctx_for_rank, winner, loser, lr=0.15)
                                    if hasattr(ranker, "save"):
                                        ranker.save()
                                elif neg and not pos:
                                    ranker.update_pair(ctx_for_rank, loser, winner, lr=0.10)
                                    if hasattr(ranker, "save"):
                                        ranker.save()
                    except Exception:
                        pass
        except Exception:
            pass

        assistant_text_override: Optional[str] = None
        final_pack_override: Optional[Dict[str, Any]] = None
        selected_macro_override = None

        if t.startswith("j'aime") and "inbox/" in t:
            import re as _re
            m = _re.search(r"(?:\"([^\"]+)\"|'([^']+)')", msg)
            if m:
                voice.update_from_liked_source(m.group(1) or m.group(2))

            paths = _re.findall(r"(inbox\/[^\s]+)", msg)
            if paths and ingest_inbox_paths:
                try:
                    added = ingest_inbox_paths(paths, arch=arch)
                except Exception:
                    added = 0
                assistant_text_override = (
                    f"Bien reçu : j’ai intégré {added} source(s) de l’inbox et capté des formules réutilisables."
                )
                final_pack_override = {
                    "text": assistant_text_override,
                    "chosen": {"text": assistant_text_override},
                    "alts": [],
                }

        if assistant_text_override is None:
            try:
                assistant_text_brut = auto.step(user_msg=msg)
            except Exception as e:
                print("⚠️ Erreur durant le cycle :", e)
                traceback.print_exc()
                continue
        else:
            assistant_text_brut = None

        reply = None
        final_pack: Optional[Dict[str, Any]] = final_pack_override
        selected_macro = selected_macro_override
        try:
            try:
                ctx = arch.context_builder.build(msg)
            except Exception:
                ctx = {"last_message": msg}

            ctx.setdefault("last_user_msg", msg)

            macro_selector = getattr(arch, "tactic_selector", None)
            if selected_macro is None and macro_selector and hasattr(macro_selector, "pick"):
                try:
                    selected_macro = macro_selector.pick(context=ctx)
                except Exception:
                    selected_macro = None

            generated_points: List[str] = []
            if isinstance(assistant_text_brut, dict):
                bullets = assistant_text_brut.get("bullets") if isinstance(assistant_text_brut, dict) else None
                if isinstance(bullets, list):
                    generated_points = [str(b).strip() for b in bullets if str(b).strip()]
                else:
                    text = assistant_text_brut.get("text") or assistant_text_brut.get("raw")
                    if text:
                        generated_points = [s.strip() for s in str(text).split("\n") if s.strip()]
            elif isinstance(assistant_text_brut, list):
                generated_points = [str(item).strip() for item in assistant_text_brut if str(item).strip()]
            elif assistant_text_brut is not None:
                generated_points = [
                    line.strip()
                    for line in str(assistant_text_brut or "").split("\n")
                    if line.strip()
                ] or [str(assistant_text_brut or "").strip()]

            plan = {"title": "", "bullets": generated_points}

            renderer = getattr(arch, "renderer", None)
            if assistant_text_override is None and renderer and hasattr(renderer, "render_final"):
                try:
                    final_pack = renderer.render_final(ctx, plan)
                    reply = (final_pack or {}).get("text")
                except Exception:
                    final_pack = final_pack_override

            if reply is None and renderer is not None and (
                assistant_text_override is None or assistant_text_brut is not None
            ):
                sem = {"text": assistant_text_brut}
                reply = arch.renderer.render_reply(sem, ctx)
                if reply is not None and final_pack is None:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if reply is None and assistant_text_override is not None:
                reply = assistant_text_override
                if final_pack is None:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if reply is None and assistant_text_brut is not None:
                reply = str(assistant_text_brut)
                if final_pack is None:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }

            if reply is not None:
                print(reply)
        except Exception as e:
            print("⚠️ Erreur lors du rendu :", e)
            traceback.print_exc()
            continue

        if reply is not None:
            if final_pack:
                try:
                    final_pack["text"] = reply
                    chosen = final_pack.get("chosen") or {}
                    if not isinstance(chosen, dict):
                        chosen = {"text": reply}
                    else:
                        chosen = dict(chosen)
                        chosen["text"] = reply
                    final_pack["chosen"] = chosen
                    if not isinstance(final_pack.get("alts"), list):
                        final_pack["alts"] = []
                except Exception:
                    final_pack = {
                        "text": reply,
                        "chosen": {"text": reply},
                        "alts": [],
                    }
            else:
                final_pack = {
                    "text": reply,
                    "chosen": {"text": reply},
                    "alts": [],
                }

            arch._last_candidates = final_pack
            arch._last_macro = selected_macro

            try:
                auto.arch.memory.add_memory({"kind": "interaction", "role": "assistant", "text": reply})
                arch.lexicon.add_from_text(reply, liked=False)
                arch.lexicon.save()
            except Exception:
                pass

        # ==== QUESTIONS PROACTIVES ====
        questions = auto.pending_questions()
        if questions:
            _pending_cache = list(questions)
        for q in questions:
            print("❓", q["text"])

if __name__ == "__main__":
    run_cli()
