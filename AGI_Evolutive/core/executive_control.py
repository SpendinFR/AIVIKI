# core/executive_control.py
import time
class ExecutiveControl:
    def __init__(self, arch):
        self.arch = arch

    def run_step(self, user_msg=None, inbox_docs=None):
        return self.arch.cycle(user_msg=user_msg, inbox_docs=inbox_docs)