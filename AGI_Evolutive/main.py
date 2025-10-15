# 🚀 main.py - Point d'entrée AGI Évolutive
import glob
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

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
from AGI_Evolutive.orchestrator import Orchestrator

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
        orc = Orchestrator(arch)
        auto = Autopilot(arch, orchestrator=orc)
    except Exception as e:
        print("❌ Erreur d'initialisation :", e)
        traceback.print_exc()
        sys.exit(1)

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
        try:
            response = auto.step(user_msg=msg)
            print("\n🤖", response)
        except Exception as e:
            print("⚠️ Erreur durant le cycle :", e)
            traceback.print_exc()
            continue

        # ==== QUESTIONS PROACTIVES ====
        questions = auto.pending_questions()
        if questions:
            _pending_cache = list(questions)
        for q in questions:
            print("❓", q["text"])

if __name__ == "__main__":
    run_cli()
