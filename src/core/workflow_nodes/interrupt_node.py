"""
Interrupt control node for LangGraph workflows.
"""

from typing import TYPE_CHECKING

from ..base_processor import ProcessorState

if TYPE_CHECKING:
    from ..base_processor import BaseProcessor


class InterruptNode:
    """
    Interrupt control node for UI-controlled interruptions.
    
    Checks for UI interrupt signals, handles checkpoint creation/restoration,
    executes user's interrupt handling logic, and integrates with LangGraph's interrupt system.
    """
    
    def __init__(self, processor: "BaseProcessor"):
        """Initialize with processor instance."""
        self.processor = processor
    
    def check_interrupt(self, state: ProcessorState) -> ProcessorState:
        """
        Check if processing should be interrupted by UI request.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state
        """
        # Check for interrupt signal
        interrupt_requested = state.get("interrupt_requested", False)
        
        if interrupt_requested:
            # Set save point ID from config
            interrupt_config = self.processor.get_capability_config("interruptible")
            save_point_id = interrupt_config.get("save_point_id", "default_save_point")
            state["save_point_id"] = save_point_id
            state["status"] = "interrupt_requested"
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(
                    state, 
                    "interrupt_detected", 
                    {"save_point_id": save_point_id}
                )
        else:
            state["status"] = "continuing"
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(state, "interrupt_check_passed")
        
        return state
    
    def handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """
        Handle UI interrupt request.
        
        Args:
            state: Current processor state with interrupt_requested=True
            
        Returns:
            Updated processor state with interrupt handling
        """
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(state, "interrupt_handling_started")
        
        try:
            # Execute user's interrupt handling logic
            updated_state = self.processor._handle_interrupt(state)
            
            # Ensure status is set appropriately
            if "status" not in updated_state or updated_state["status"] == "interrupt_requested":
                updated_state["status"] = "interrupted"
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(
                    updated_state, 
                    "interrupt_handling_completed", 
                    {"final_status": updated_state.get("status")}
                )
            
            return updated_state
            
        except Exception as interrupt_error:
            # Interrupt handling itself failed
            state["interrupt_error"] = interrupt_error
            state["status"] = "interrupt_handling_failed"
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(
                    state, 
                    "interrupt_handling_failed", 
                    {"interrupt_error": str(interrupt_error)}
                )
            
            return state
    
    def create_checkpoint(self, state: ProcessorState) -> ProcessorState:
        """
        Create checkpoint for later resumption.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state with checkpoint information
        """
        save_point_id = state.get("save_point_id", "default_save_point")
        
        # In real implementation, this would save state to persistent storage
        # For now, we just mark it in the state
        state["checkpoint_created"] = True
        state["checkpoint_timestamp"] = "2024-01-01T00:00:00Z"  # Placeholder
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(
                state, 
                "checkpoint_created", 
                {"save_point_id": save_point_id}
            )
        
        return state
    
    def resume_from_checkpoint(self, save_point_id: str) -> ProcessorState:
        """
        Resume processing from a saved checkpoint.
        
        Args:
            save_point_id: ID of the save point to resume from
            
        Returns:
            Restored processor state
        """
        # In real implementation, this would load state from persistent storage
        # For now, we create a basic restored state
        restored_state = ProcessorState(
            input_data=None,  # Would be restored from checkpoint
            output_data=None,
            error=None,
            status="resumed_from_checkpoint",
            retry_count=0,
            interrupt_requested=False,
            save_point_id=save_point_id
        )
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(
                restored_state, 
                "resumed_from_checkpoint", 
                {"save_point_id": save_point_id}
            )
        
        return restored_state
    
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
            "node_name": "interrupt_control",
            "processor_type": self.processor.get_processor_type().value,
            "state": {
                "status": state.get("status"),
                "interrupt_requested": state.get("interrupt_requested", False),
                "save_point_id": state.get("save_point_id")
            }
        }
        
        if extra_data:
            event_data.update(extra_data)
        
        # Send to each observer
        for observer in observers:
            print(f"[MONITOR:{observer}] {event_data}")  # Placeholder implementation 