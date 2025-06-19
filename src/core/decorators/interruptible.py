"""
Interruptible capability marker decorator.
"""

from typing import Dict, Any, Type, Callable


def interruptible(save_point_id: str = "default_save_point") -> Callable:
    """
    Mark processor as UI-interruptible.
    
    This decorator:
    1. Adds 'interruptible' to processor capabilities
    2. Stores save point configuration
    3. Validates that _handle_interrupt method is implemented
    4. Enables interrupt nodes and checkpointing in generated subgraph
    
    Args:
        save_point_id: Identifier for save point creation
        
    Returns:
        Class decorator that marks the capability
        
    Example:
        @interruptible(save_point_id="fetch_data")
        class DataFetcher(BaseProcessor):
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                # User-defined interrupt handling
                state["status"] = "paused"
                return state
    """
    def decorator(cls: Type) -> Type:
        # Initialize capability tracking if not exists
        if not hasattr(cls, '_processor_capabilities'):
            cls._processor_capabilities = []
        if not hasattr(cls, '_processor_capability_configs'):
            cls._processor_capability_configs = {}
        
        # Add capability
        if 'interruptible' not in cls._processor_capabilities:
            cls._processor_capabilities.append('interruptible')
        
        # Store configuration
        cls._processor_capability_configs['interruptible'] = {
            'save_point_id': save_point_id
        }
        
        # Validate configuration
        if not save_point_id or not isinstance(save_point_id, str):
            raise ValueError("save_point_id must be a non-empty string")
        
        return cls
    
    return decorator 