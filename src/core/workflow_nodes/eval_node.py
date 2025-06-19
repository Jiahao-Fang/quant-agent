"""
Evaluation node for LangGraph workflows.
"""

from typing import TYPE_CHECKING

from ..base_processor import ProcessorState

if TYPE_CHECKING:
    from ..base_processor import BaseProcessor


class EvalNode:
    """
    Evaluation node that wraps _evaluate_result.
    
    Executes user's evaluation logic, determines pass/fail status,
    and manages evaluation retry logic.
    """
    
    def __init__(self, processor: "BaseProcessor"):
        """Initialize with processor instance."""
        self.processor = processor
    
    def execute(self, state: ProcessorState) -> ProcessorState:
        """
        Execute evaluation logic for processing results.
        
        Args:
            state: Current processor state with output_data
            
        Returns:
            Updated processor state with evaluation results
        """
        # Increment eval retry count
        eval_retry_count = state.get("eval_retry_count", 0)
        state["eval_retry_count"] = eval_retry_count + 1
        
        # Store eval config in state
        eval_config = self.processor.get_capability_config("evaluable")
        state["eval_config"] = eval_config
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(state, "evaluation_started")
        
        try:
            # Execute user's evaluation logic
            updated_state = self.processor._evaluate_result(state)
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                eval_passed = updated_state.get("eval_passed", False)
                self._send_monitoring_event(
                    updated_state, 
                    "evaluation_completed", 
                    {"eval_passed": eval_passed}
                )
            
            return updated_state
            
        except Exception as eval_error:
            # Evaluation itself failed - treat as eval failure
            state["eval_error"] = eval_error
            state["eval_passed"] = False
            
            # Send monitoring event if observable
            if self.processor.has_capability("observable"):
                self._send_monitoring_event(
                    state, 
                    "evaluation_failed", 
                    {"eval_error": str(eval_error)}
                )
            
            return state
    
    def check_retry(self, state: ProcessorState) -> ProcessorState:
        """
        Check if retry should be attempted after evaluation failure.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state
        """
        eval_config = state.get("eval_config", {})
        max_retries = eval_config.get("max_retries", 3)
        eval_retry_count = state.get("eval_retry_count", 0)
        
        can_retry = (
            not state.get("eval_passed", False) and 
            eval_retry_count < max_retries
        )
        
        if can_retry:
            # Clear previous results for retry
            state["output_data"] = None
            state["eval_passed"] = False
            state["status"] = "retrying_after_eval"
        else:
            # No more retries
            state["status"] = "failed_after_eval"
        
        # Send monitoring event if observable
        if self.processor.has_capability("observable"):
            self._send_monitoring_event(
                state, 
                "eval_retry_decision", 
                {"can_retry": can_retry, "retry_count": eval_retry_count}
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
            "node_name": "evaluate_result",
            "processor_type": self.processor.get_processor_type().value,
            "state": {
                "status": state.get("status"),
                "eval_retry_count": state.get("eval_retry_count", 0),
                "eval_passed": state.get("eval_passed", False)
            }
        }
        
        if extra_data:
            event_data.update(extra_data)
        
        # Send to each observer
        for observer in observers:
            print(f"[MONITOR:{observer}] {event_data}")  # Placeholder implementation 