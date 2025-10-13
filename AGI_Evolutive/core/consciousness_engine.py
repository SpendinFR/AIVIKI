# core/consciousness_engine.py
from .global_workspace import GlobalWorkspace
class ConsciousnessEngine:
    def __init__(self):
        self.gw = GlobalWorkspace()

    def push(self, content):
        self.gw.broadcast(content)