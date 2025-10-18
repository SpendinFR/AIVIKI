"""Compatibility layer for relocated memory utilities.

The progressive summarizer and semantic memory manager now live under
``AGI_Evolutive.memory``. Importing from ``memory`` continues to work by
forwarding to the new package.
"""

try:
    from AGI_Evolutive.memory.semantic_memory_manager import (
        SemanticMemoryManager,
    )
    from AGI_Evolutive.memory.summarizer import (
        ProgressiveSummarizer,
        SummarizerConfig,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for legacy layouts
    from .semantic_memory_manager import SemanticMemoryManager
    from .summarizer import ProgressiveSummarizer, SummarizerConfig
"""
Wire up the improved SemanticMemoryManager within your memory system.
This is an example snippet; adapt to your project structure.
"""

__all__ = [
    "SemanticMemoryManager",
    "ProgressiveSummarizer",
    "SummarizerConfig",
]
