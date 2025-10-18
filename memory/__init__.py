"""Compatibility layer for relocated memory utilities.

The progressive summarizer and semantic memory manager now live under
``AI.AGI_Evolutive.memory``. Importing from ``AI.memory`` continues to work by
forwarding to the new package.
"""

from AI.AGI_Evolutive.memory.semantic_memory_manager import (
    SemanticMemoryManager,
)
from AI.AGI_Evolutive.memory.summarizer import (
    ProgressiveSummarizer,
    SummarizerConfig,
)

__all__ = [
    "SemanticMemoryManager",
    "ProgressiveSummarizer",
    "SummarizerConfig",
]
