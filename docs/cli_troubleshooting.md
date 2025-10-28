# AGI Évolutive CLI – Démarrage & Dépannage

La CLI d'AGI Évolutive charge une architecture cognitive très complète. Sur
certaines machines, surtout lors du premier lancement, la séquence
d'initialisation peut durer plusieurs dizaines de secondes pendant
l'importation des modèles et la mise en place des sous-systèmes. Pendant cette
phase, aucune invite n'est affichée et les commandes comme `/quit` ne sont pas
encore disponibles : il faut attendre l'apparition du message `✅ AGI initialisée`
ainsi que du prompt `> `.

## Options utiles

La commande `python -m AGI_Evolutive.main` accepte désormais plusieurs options
pour adapter le démarrage :

* `--boot-minimal` : démarre l'architecture en mode réduit pour obtenir une
  invite rapidement. Certaines fonctionnalités avancées (RAG, auto-amélioration,
  etc.) restent désactivées tant que vous n'avez pas relancé sans ce drapeau.
* `--boot-progress` : affiche les étapes internes du chargement (mémoire,
  perception, métacognition…). Pratique pour vérifier que l'initialisation
  progresse toujours et repérer un éventuel blocage.
* `--no-auto-llm` : empêche l'activation automatique de l'intégration LLM si
  vous souhaitez tout contrôler manuellement ou travailler hors connexion.

Exemples :

```bash
# Démarrage rapide pour un test
python -m AGI_Evolutive.main --boot-minimal --boot-progress

# Session complète mais sans activer automatiquement le LLM
python -m AGI_Evolutive.main --no-auto-llm
```

## Astuces supplémentaires

* Le premier lancement peut être plus long, car certains modèles (ex. vecteurs
  sémantiques) sont générés ou téléchargés. Les exécutions suivantes seront
  nettement plus rapides.
* Si la fenêtre console semble figée, vérifiez qu'aucune sélection de texte n'est
  active (mode « QuickEdit » sur Windows). Dans ce mode, l'entrée clavier est
  temporairement suspendue.
* `Ctrl+C` force une interruption propre. L'agent déclenche alors une sauvegarde
  et se ferme.

Ces options offrent un retour visuel immédiat et permettent d'interagir avec la
CLI même sur un matériel plus limité.
