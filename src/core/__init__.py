"""
Core module for AI Quant Agent system.

Provides base processor class, dynamic LangGraph subgraph generation,
and capability-based decorators for building modular processors.
"""

# Core abstractions
from .base_processor import BaseProcessor, ProcessorType, ProcessorState, ProcessorResult

# Dynamic subgraph generation
from .subgraph_builder import SubgraphBuilder

# Factory and coordination
from .processor_factory import ProcessorFactory
from .pipeline_coordinator import PipelineCoordinator

# Capability decorators
from .decorators import observable, evaluable, debuggable, interruptible

# Workflow nodes
from .workflow_nodes import ProcessNode, DebugNode, EvalNode, InterruptNode

__all__ = [
    # Core abstractions
    "BaseProcessor",
    "ProcessorType", 
    "ProcessorState",
    "ProcessorResult",
    
    # Subgraph generation
    "SubgraphBuilder",
    
    # Factory and coordination
    "ProcessorFactory",
    "PipelineCoordinator",
    
    # Capability decorators
    "observable",
    "evaluable",
    "debuggable", 
    "interruptible",
    
    # Workflow nodes
    "ProcessNode",
    "DebugNode",
    "EvalNode", 
    "InterruptNode"
] 