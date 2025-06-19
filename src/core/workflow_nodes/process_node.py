"""
Core processing node for LangGraph workflows.
"""

from typing import TYPE_CHECKING

from ..base_processor import ProcessorState

if TYPE_CHECKING:
    from ..base_processor import BaseProcessor


class ProcessNode:
    """
    Standard processing node that wraps _process_core_logic.
    
    Handles exceptions and state updates, integrates with LangGraph state management.
    """
    
    def __init__(self, processor: "BaseProcessor"):
        """Initialize with processor instance."""
        self.processor = processor
    
    def execute(self, state: ProcessorState) -> ProcessorState:
        """
        Execute the core processing logic.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state
        """
        try:
            # Clear any previous errors
            state["error"] = None
            state["status"] = "processing"
            
            # Execute core logic
            updated_state = self.processor._process_core_logic(state)
            
            # Mark as successful if no exception
            updated_state["status"] = "success"
            
            # Send monitoring events if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(updated_state, "process_completed")
            
            return updated_state
            
        except Exception as e:
            # Capture error for debug capability
            state["error"] = e
            state["status"] = "error"
            
            # Send monitoring events if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(state, "process_failed", {"error": str(e)})
            
            return state
    
    def _send_monitoring_event(
        self, 
        state: ProcessorState, 
        event_type: str, 
        extra_data: dict = None
    ) -> None:
        """Send monitoring event to observers."""
        if not self.processor.has_capability("observable"):
            return
        
        config = self.processor.get_capability_config("observable")
        observers = config.get("observers", [])
        
        event_data = {
            "event_type": event_type,
            "node_name": "process_core",
            "processor_type": self.processor.get_processor_type().value,
            "state": {
                "status": state.get("status"),
                "retry_count": state.get("retry_count", 0),
                "has_output": state.get("output_data") is not None
            }
        }
        
        if extra_data:
            event_data.update(extra_data)
        
        # Send to each observer
        # Note: In real implementation, this would use proper observer pattern
        for observer in observers:
            print(f"[MONITOR:{observer}] {event_data}")  # Placeholder implementation 