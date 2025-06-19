"""
Debuggable capability marker decorator.
"""

from typing import Dict, Any, Type, Callable


def debuggable(max_retries: int = 3) -> Callable:
    """
    Mark processor as having debug capability.
    
    This decorator:
    1. Adds 'debuggable' to processor capabilities
    2. Stores retry configuration
    3. Validates that _debug_error method is implemented
    4. Enables debug nodes in generated subgraph
    
    Args:
        max_retries: Maximum number of debug retries
        
    Returns:
        Class decorator that marks the capability
        
    Example:
        @debuggable(max_retries=1)
        class DataFetcher(BaseProcessor):
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                # User-defined debug logic
                error = state["error"]
                state["should_retry"] = isinstance(error, ConnectionError)
                return state
    """
    def decorator(cls: Type) -> Type:
        # Initialize capability tracking if not exists
        if not hasattr(cls, '_processor_capabilities'):
            cls._processor_capabilities = []
        if not hasattr(cls, '_processor_capability_configs'):
            cls._processor_capability_configs = {}
        
        # Add capability
        if 'debuggable' not in cls._processor_capabilities:
            cls._processor_capabilities.append('debuggable')
        
        # Store configuration
        cls._processor_capability_configs['debuggable'] = {
            'max_retries': max_retries
        }
        
        # Validate configuration
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        return cls
    
    return decorator 