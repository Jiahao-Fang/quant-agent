"""
Observable capability marker decorator.
"""

from typing import List, Dict, Any, Type, Callable
from functools import wraps


def observable(observers: List[str]) -> Callable:
    """
    Mark processor as having monitoring capability.
    
    This decorator:
    1. Adds 'observable' to processor capabilities
    2. Stores observer configuration
    3. Validates no additional methods required
    4. Enables monitoring nodes in generated subgraph
    
    Args:
        observers: List of observer names to notify
        
    Returns:
        Class decorator that marks the capability
        
    Example:
        @observable(observers=["ui", "logger"])
        class DataFetcher(BaseProcessor):
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                # Business logic only
                return state
    """
    def decorator(cls: Type) -> Type:
        # Initialize capability tracking if not exists
        if not hasattr(cls, '_processor_capabilities'):
            cls._processor_capabilities = []
        if not hasattr(cls, '_processor_capability_configs'):
            cls._processor_capability_configs = {}
        
        # Add capability
        if 'observable' not in cls._processor_capabilities:
            cls._processor_capabilities.append('observable')
        
        # Store configuration
        cls._processor_capability_configs['observable'] = {
            'observers': observers
        }
        
        # Validate configuration
        if not observers:
            raise ValueError("Observable decorator requires at least one observer")
        
        if not all(isinstance(obs, str) for obs in observers):
            raise ValueError("All observers must be strings")
        
        return cls
    
    return decorator 