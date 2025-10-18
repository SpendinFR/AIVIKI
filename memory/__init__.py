"""
Wire up the improved SemanticMemoryManager within your memory system.
This is an example snippet; adapt to your project structure.
"""

from .semantic_memory_manager import SemanticMemoryManager
from .summarizer import ProgressiveSummarizer, SummarizerConfig

__all__ = [
    "SemanticMemoryManager",
    "ProgressiveSummarizer",
    "SummarizerConfig",
]
