# 🚀 main.py — Point d’entrée AGI Évolutive
import os, sys, glob, time, traceback
from core.cognitive_architecture import CognitiveArchitecture
from core.autopilot import Autopilot

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
  /state       → montre les infos d’état globales
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
    print("Chargement de l’architecture cognitive…")
    try:
        arch = CognitiveArchitecture()
        auto = Autopilot(arch)
    except Exception as e:
        print("❌ Erreur d’initialisation :", e)
        traceback.print_exc()
        sys.exit(1)

    print("✅ AGI initialisée. (Persistance & mémoire prêtes)")
    print(HELP_TEXT)
    print("🗨️  Démarrez la conversation ou tapez /help.")

    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⏳ Sauvegarde avant sortie…")
            try:
                auto.save_now()
            except Exception as e:
                print("⚠️ Erreur lors de la sauvegarde :", e)
            print("👋 Fin de session.")
            break

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
                print("⚠️ Impossible d’afficher l’état :", e)
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
        for q in questions:
            print("❓", q["text"])

if __name__ == "__main__":
    run_cli()
