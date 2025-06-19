"""
Capability marking decorators for processors.

These decorators mark processor capabilities and store configuration
for dynamic LangGraph subgraph generation.
"""

from .observable import observable
from .evaluable import evaluable
from .debuggable import debuggable
from .interruptible import interruptible

__all__ = [
    "observable",
    "evaluable", 
    "debuggable",
    "interruptible"
] 