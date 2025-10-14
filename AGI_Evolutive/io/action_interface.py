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
