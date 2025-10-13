# core/global_workspace.py
class GlobalWorkspace:
    def __init__(self):
        self.broadcasts = []

    def broadcast(self, item):
        self.broadcasts.append(item)
        if len(self.broadcasts) > 1000:
            self.broadcasts.pop(0)