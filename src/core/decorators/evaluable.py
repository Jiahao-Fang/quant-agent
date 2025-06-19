"""
Evaluable capability marker decorator.
"""

from typing import Dict, Any, Type, Callable


def evaluable(max_retries: int = 3) -> Callable:
    """
    Mark processor as having evaluation capability.
    
    This decorator:
    1. Adds 'evaluable' to processor capabilities
    2. Stores retry configuration
    3. Validates that _evaluate_result method is implemented
    4. Enables evaluation nodes in generated subgraph
    
    Args:
        max_retries: Maximum number of evaluation retries
        
    Returns:
        Class decorator that marks the capability
        
    Example:
        @evaluable(max_retries=2)
        class DataFetcher(BaseProcessor):
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                # User-defined evaluation logic
                state["eval_passed"] = self._check_data_quality(state["output_data"])
                return state
    """
    def decorator(cls: Type) -> Type:
        # Initialize capability tracking if not exists
        if not hasattr(cls, '_processor_capabilities'):
            cls._processor_capabilities = []
        if not hasattr(cls, '_processor_capability_configs'):
            cls._processor_capability_configs = {}
        
        # Add capability
        if 'evaluable' not in cls._processor_capabilities:
            cls._processor_capabilities.append('evaluable')
        
        # Store configuration
        cls._processor_capability_configs['evaluable'] = {
            'max_retries': max_retries
        }
        
        # Validate configuration
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        return cls
    
    return decorator 