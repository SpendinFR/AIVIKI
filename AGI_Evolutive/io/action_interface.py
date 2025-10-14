import os, json, time, uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List


def _now():
    return time.time()


@dataclass
class Action:
    id: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5         # 0..1
    created_at: float = field(default_factory=_now)
    status: str = "queued"        # queued | running | done | failed | skipped
    result: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ActionInterface:
    """
    - Maintient une file d’actions (internes + issues de Goals/Planner)
    - Valide contre policy (si présent)
    - Exécute en vrai (si pont dispo) ou simule
    - Journalise dans data/actions_log.jsonl + ajoute une mémoire d’expérience
    - Informe le système émotionnel & métacognitif
    """

    def __init__(self,
                 path_log: str = "data/actions_log.jsonl",
                 output_dir: str = "data/output"):
        os.makedirs(os.path.dirname(path_log), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.path_log = path_log
        self.output_dir = output_dir
        self.queue: List[Action] = []
        self.bound = {
            "arch": None, "goals": None, "policy": None,
            "memory": None, "metacog": None, "emotions": None, "language": None
        }
        self.cooldown_s = 0.0  # peut être modulé par l’émotion

    # ---------- binding ----------
    def bind(self, arch=None, goals=None, policy=None,
             memory=None, metacog=None, emotions=None, language=None):
        self.bound.update({
            "arch": arch, "goals": goals, "policy": policy,
            "memory": memory, "metacog": metacog, "emotions": emotions, "language": language
        })

    # ---------- file d’attente ----------
    def enqueue(self, type_: str, payload: Dict[str, Any], priority: float = 0.5, context: Dict[str, Any] = None) -> str:
        act = Action(id=str(uuid.uuid4()), type=type_, payload=payload or {}, priority=float(priority),
                     context=context or {})
        self.queue.append(act)
        # tri simple par priority (desc)
        self.queue.sort(key=lambda a: a.priority, reverse=True)
        return act.id

    def _pull_from_goals(self) -> Optional[Action]:
        """Essaye de tirer la prochaine action depuis le GoalSystem si dispo."""
        goals = self.bound.get("goals")
        try:
            if goals and hasattr(goals, "pop_next_action"):
                nxt = goals.pop_next_action()
                if nxt:
                    return Action(id=str(uuid.uuid4()), type=nxt.get("type", "reflect"),
                                  payload=nxt.get("payload", {}), priority=float(nxt.get("priority", 0.5)),
                                  context={"source": "goals"})
            elif goals and hasattr(goals, "get_next_action"):
                nxt = goals.get_next_action()
                if nxt:
                    return Action(id=str(uuid.uuid4()), type=nxt.get("type", "reflect"),
                                  payload=nxt.get("payload", {}), priority=float(nxt.get("priority", 0.5)),
                                  context={"source": "goals"})
        except Exception:
            pass
        return None

    # ---------- cycle ----------
    def step(self):
        # cooldown modulé par état émotionnel
        emo = self.bound.get("emotions")
        if emo and hasattr(emo, "get_emotional_modulators"):
            mods = emo.get_emotional_modulators() or {}
            exploration = float(mods.get("exploration_rate", 0.15))
            # plus d’exploration → moins de cooldown
            self.cooldown_s = max(0.0, 1.0 - 0.8 * exploration)
        else:
            self.cooldown_s = 0.5

        # 1) si rien en file → tirer depuis Goals
        if not self.queue:
            pulled = self._pull_from_goals()
            if pulled:
                self.queue.append(pulled)

        # 2) si toujours vide → générer une micro-action autonome
        if not self.queue:
            self._maybe_autonomous_microaction()

        # 3) exécuter au plus une action
        if not self.queue:
            return
        act = self.queue.pop(0)
        self._execute(act)
        time.sleep(self.cooldown_s)

    # ---------- exécution ----------
    def _execute(self, act: Action):
        policy = self.bound.get("policy")
        arch = self.bound.get("arch")
        memory = self.bound.get("memory")
        metacog = self.bound.get("metacog")
        emotions = self.bound.get("emotions")

        # policy check (si dispo)
        if policy and hasattr(policy, "validate_action"):
            try:
                ok, reason = policy.validate_action(asdict(act))
                if not ok:
                    act.status = "skipped"
                    act.result = {"ok": False, "reason": reason or "policy_rejected"}
                    self._log(act)
                    self._memorize_action(act)
                    return
            except Exception:
                pass

        # mapping des handlers
        handlers = {
            "message_user": self._h_message_user,
            "write_memory": self._h_write_memory,
            "save_file":    self._h_save_file,
            "reflect":      self._h_reflect,
            "learn_concept":self._h_learn_concept,
            "search_memory":self._h_search_memory,
        }
        handler = handlers.get(act.type, self._h_simulate)

        act.status = "running"
        try:
            result = handler(act)
            act.status = "done" if (isinstance(result, dict) and result.get("ok") is not False) else "failed"
            act.result = result if isinstance(result, dict) else {"ok": True, "data": result}
        except Exception as e:
            act.status = "failed"
            act.result = {"ok": False, "error": str(e)}

        # reward sommaire + events émotion/métacog
        reward = self._shape_reward(act)
        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(
                    kind="action_success" if act.status == "done" else "action_failure",
                    intensity=0.4 if act.status == "done" else 0.6,
                    valence_hint=+0.4 if act.status == "done" else -0.5,
                    arousal_hint=0.2,
                    meta={"action_type": act.type}
                )
            except Exception:
                pass

        if metacog and hasattr(metacog, "_record_metacognitive_event"):
            try:
                metacog._record_metacognitive_event(
                    event_type="action_executed",
                    domain=getattr(metacog, "CognitiveDomain", None) or getattr(metacog, "CognitiveDomain", None),
                    description=f"Action {act.type} -> {act.status}",
                    significance=0.3 if act.status == "done" else 0.5,
                    confidence=0.7,
                    emotional_valence=+0.3 if act.status == "done" else -0.2,
                    cognitive_load=0.2
                )
            except Exception:
                pass

        self._log(act, reward=reward)
        self._memorize_action(act, reward=reward)

    # ---------- handlers ----------
    def _h_message_user(self, act: Action) -> Dict[str, Any]:
        """
        Simule l'envoi d'un message : on écrit dans output/last_message.txt + on retourne le texte.
        Si un module Language existe, on peut l’utiliser pour générer le texte final à partir d’un intent.
        """
        lang = self.bound.get("language")
        text = act.payload.get("text")
        if not text and lang and hasattr(lang, "generate"):
            # fallback : génération
            intent = act.payload.get("intent", "inform")
            text = lang.generate({"intent": intent, "hints": getattr(lang, "style_hints", {})})
        text = text or "(message vide)"

        path = os.path.join(self.output_dir, "last_message.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": path, "text": text}

    def _h_write_memory(self, act: Action) -> Dict[str, Any]:
        memory = self.bound.get("memory")
        if not (memory and hasattr(memory, "add_memory")):
            return {"ok": False, "reason": "memory_unavailable"}
        kind = act.payload.get("kind", "note")
        content = act.payload.get("content", "")
        meta = act.payload.get("meta", {})
        try:
            mem_id = memory.add_memory(kind=kind, content=content, metadata=meta)
            return {"ok": True, "memory_id": mem_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_save_file(self, act: Action) -> Dict[str, Any]:
        name = act.payload.get("name", f"artifact_{int(time.time())}.txt")
        content = act.payload.get("content", "")
        path = os.path.join(self.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": path}

    def _h_reflect(self, act: Action) -> Dict[str, Any]:
        metacog = self.bound.get("metacog")
        if not (metacog and hasattr(metacog, "trigger_reflection")):
            return {"ok": False, "reason": "metacog_unavailable"}
        trig = act.payload.get("trigger", "action_reflect")
        domain = getattr(getattr(metacog, "CognitiveDomain", None), "REASONING", None)
        try:
            ref = metacog.trigger_reflection(trigger=trig, domain=domain, urgency=0.4, depth=2)
            return {"ok": True, "reflection": {"duration": ref.duration, "quality": ref.quality_score}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_learn_concept(self, act: Action) -> Dict[str, Any]:
        """
        Enregistre un objectif d’apprentissage + mémoire associée.
        """
        goals = self.bound.get("goals")
        memory = self.bound.get("memory")
        concept = act.payload.get("concept", "concept_inconnu")
        why = act.payload.get("why", "lacune détectée")
        try:
            if goals and hasattr(goals, "add_learning_goal"):
                goals.add_learning_goal(concept, motive=why, expected_gain=0.4)
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="learning_intent",
                    content=f"Apprendre le concept: {concept}",
                    metadata={"reason": why, "source": "action_interface"}
                )
            return {"ok": True, "concept": concept}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_search_memory(self, act: Action) -> Dict[str, Any]:
        memory = self.bound.get("memory")
        query = act.payload.get("query", "")
        if not query:
            return {"ok": False, "reason": "empty_query"}
        try:
            # support : search() ou get_recent_memories()
            if hasattr(memory, "search"):
                hits = memory.search(query, top_k=10)
            else:
                hits = (memory.get_recent_memories(n=50) or [])
                hits = [m for m in hits if query.lower() in (str(m.get("content", "")).lower())]
            return {"ok": True, "hits": hits[:10]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_simulate(self, act: Action) -> Dict[str, Any]:
        """
        Fallback générique : simule action avec succès ~80%.
        """
        ok = (time.time()*1000) % 10 != 0  # pseudo aléa
        return {"ok": bool(ok), "simulated": True, "type": act.type}

    # ---------- logging & mémoire ----------
    def _log(self, act: Action, reward: float = 0.0):
        rec = asdict(act)
        rec["reward"] = reward
        rec["logged_at"] = _now()
        line = json.dumps(rec, ensure_ascii=False)
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _memorize_action(self, act: Action, reward: float = 0.0):
        memory = self.bound.get("memory")
        if not (memory and hasattr(memory, "add_memory")):
            return
        try:
            memory.add_memory(
                kind="action_experience",
                content=f"[{act.type}] -> {act.status}",
                metadata={
                    "action_id": act.id, "payload": act.payload,
                    "result": act.result, "reward": reward,
                    "priority": act.priority, "created_at": act.created_at
                }
            )
        except Exception:
            pass

    def _shape_reward(self, act: Action) -> float:
        """
        Reward heuristique : success + ton émotionnel + utilité perçue.
        """
        base = 0.6 if act.status == "done" else -0.5
        util = 0.2 if act.type in ("learn_concept", "reflect", "write_memory") else 0.0
        return float(max(-1.0, min(1.0, base + util)))

    # ---------- micro-actions autonomes ----------
    def _maybe_autonomous_microaction(self):
        """
        Génère une petite action autonome (réflexion / scan mémoire)
        quand il n’y a rien à faire → l’agent reste vivant.
        """
        emo = self.bound.get("emotions")
        curiosity = 0.2
        if emo and hasattr(emo, "get_emotional_modulators"):
            mods = emo.get_emotional_modulators() or {}
            curiosity = float(mods.get("curiosity_gain", 0.2)) + float(mods.get("exploration_rate", 0.15))
        # seuil simple
        if curiosity > 0.25:
            # alterner entre réflexion et recherche mémoire
            if int(time.time()) % 2 == 0:
                self.enqueue("reflect", {"trigger": "idle_reflection"}, priority=0.55, context={"auto": True})
            else:
                self.enqueue("search_memory", {"query": "lacune|erreur|incompréhension"}, priority=0.52, context={"auto": True})
import os, json, time
from typing import Dict, Any, Callable

class ActionInterface:
    """
    Exécute concrètement une action planifiée.
    Handlers extensibles: register_handler(type, fn)
    Fallback: simulation + log en mémoire.
    """
    def __init__(self, memory_store, output_dir: str = "data/outputs"):
        self.memory = memory_store
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.handlers: Dict[str, Callable[[Dict[str,Any]], Dict[str,Any]]] = {}

    def register_handler(self, action_type: str, fn: Callable[[Dict[str,Any]], Dict[str,Any]]):
        self.handlers[action_type] = fn

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        a_type = (action.get("type") or "simulate").lower()
        if a_type in self.handlers:
            try:
                res = self.handlers[a_type](action)
                self._log(action, res, ok=True)
                return res
            except Exception as e:
                res = {"ok": False, "error": str(e)}
                self._log(action, res, ok=False)
                return res
        # Fallbacks
        if a_type == "write_file":
            path = os.path.join(self.output_dir, action.get("filename","out.txt"))
            with open(path, "w", encoding="utf-8") as f:
                f.write(action.get("content",""))
            res = {"ok": True, "path": path}
        elif a_type == "communicate":
            # ici tu peux plugger un connecteur (discord, slack, etc.)
            res = {"ok": True, "echo": action.get("text","" )}
        else:
            res = {"ok": True, "simulated": True}
        self._log(action, res, ok=res.get("ok", False))
        return res

    def _log(self, action: Dict[str,Any], result: Dict[str,Any], ok: bool):
        self.memory.add_memory({
            "kind": "action_exec" if ok else "error_action",
            "text": f"ACTION {action.get('type','simulate')}",
            "action": action, "result": result, "ts": time.time()
        })
