"""
Debug node for LangGraph workflows.
"""

from typing import TYPE_CHECKING

from ..base_processor import ProcessorState

if TYPE_CHECKING:
    from ..base_processor import BaseProcessor


class DebugNode:
    """
    Debug node that wraps _debug_error for error handling.
    
    Activates when processing errors occur, executes user's debug logic,
    and manages retry decisions and state transitions.
    """
    
    def __init__(self, processor: "BaseProcessor"):
        """Initialize with processor instance."""
        self.processor = processor
    
    def execute(self, state: ProcessorState) -> ProcessorState:
        """
        Execute debug logic for processing errors.
        
        Args:
            state: Current processor state with error set
            
        Returns:
            Updated processor state with debug results
        """
        # Increment debug retry count
        debug_retry_count = state.get("debug_retry_count", 0)
        state["debug_retry_count"] = debug_retry_count + 1
        
        # Store debug config in state
        debug_config = self.processor.get_capability_config("debuggable")
        state["debug_config"] = debug_config
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(state, "debug_started")
        
        try:
            # Execute user's debug logic
            updated_state = self.processor._debug_error(state)
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                should_retry = updated_state.get("should_retry", False)
                self._send_monitoring_event(
                    updated_state, 
                    "debug_completed", 
                    {"should_retry": should_retry}
                )
            
            return updated_state
            
        except Exception as debug_error:
            # Debug itself failed
            state["debug_error"] = debug_error
            state["should_retry"] = False
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(
                    state, 
                    "debug_failed", 
                    {"debug_error": str(debug_error)}
                )
            
            return state
    
    def check_retry(self, state: ProcessorState) -> ProcessorState:
        """
        Check if retry should be attempted after debug.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state
        """
        debug_config = state.get("debug_config", {})
        max_retries = debug_config.get("max_retries", 3)
        debug_retry_count = state.get("debug_retry_count", 0)
        
        can_retry = (
            state.get("should_retry", False) and 
            debug_retry_count < max_retries
        )
        
        if can_retry:
            # Clear error for retry
            state["error"] = None
            state["status"] = "retrying_after_debug"
        else:
            # No more retries
            state["status"] = "failed_after_debug"
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(
                state, 
                "debug_retry_decision", 
                {"can_retry": can_retry, "retry_count": debug_retry_count}
            )
        
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
            "node_name": "debug_error",
            "processor_type": self.processor.get_processor_type().value,
            "state": {
                "status": state.get("status"),
                "debug_retry_count": state.get("debug_retry_count", 0),
                "error": str(state.get("error", "")) if state.get("error") else None
            }
        }
        
        if extra_data:
            event_data.update(extra_data)
        
        # Send to each observer
        for observer in observers:
            print(f"[MONITOR:{observer}] {event_data}")  # Placeholder implementation 