"""
Interrupt System Module
Similar to C++ exception handling and control flow
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from .pipeline_state import PipelineState, PipelineStatus


class PipelineInterrupt(Exception):
    """
    Custom exception for pipeline interruption
    Similar to C++ custom exception classes
    """
    
    def __init__(self, message: str, checkpoint_data: Optional[Dict[str, Any]] = None):
        self.message = message
        self.checkpoint_data = checkpoint_data or {}
        super().__init__(self.message)


class InterruptHandler(ABC):
    """
    Abstract base class for interrupt handlers
    Similar to C++ abstract base classes
    """
    
    @abstractmethod
    def can_interrupt(self, step_name: str, state: PipelineState) -> bool:
        """Check if interruption is allowed at this step"""
        pass
    
    @abstractmethod
    def handle_interrupt(self, step_name: str, state: PipelineState, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the interruption and return checkpoint data"""
        pass


class DefaultInterruptHandler(InterruptHandler):
    """
    Default implementation of interrupt handler
    Similar to C++ default implementations
    """
    
    def can_interrupt(self, step_name: str, state: PipelineState) -> bool:
        """Allow interruption at most steps when running"""
        return state.can_pause()
    
    def handle_interrupt(self, step_name: str, state: PipelineState, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standard interrupt handling"""
        return {
            'step_name': step_name,
            'timestamp': checkpoint_data.get('timestamp'),
            'state_snapshot': state.export_state(),
            'checkpoint_data': checkpoint_data
        }


class InterruptController:
    """
    Central controller for interrupt management
    Similar to C++ controller pattern with dependency injection
    """
    
    def __init__(self, state: PipelineState, handler: Optional[InterruptHandler] = None):
        self.state = state
        self.handler = handler or DefaultInterruptHandler()
        self.interrupt_points = {
            'demand_analysis': True,
            'data_fetching': True,
            'code_generation': True,
            'code_evaluation': True,
            'code_debugging': True,
            'final_execution': False  # Don't interrupt during final execution
        }
        self.checkpoint_callback: Optional[Callable] = None
    
    def set_checkpoint_callback(self, callback: Callable):
        """Set callback for checkpoint operations"""
        self.checkpoint_callback = callback
    
    def register_interrupt_point(self, step_name: str, allow_interrupt: bool = True):
        """Register a step as an interrupt point"""
        self.interrupt_points[step_name] = allow_interrupt
    
    def check_interrupt_point(self, step_name: str, checkpoint_data: Dict[str, Any]):
        """
        Check if pipeline should be interrupted at this point
        Similar to C++ RAII pattern with automatic resource management
        """
        # Check if interruption is requested
        if not self.state.interrupt_requested:
            return
        
        # Check if this step allows interruption
        if not self.interrupt_points.get(step_name, True):
            return
        
        # Check if handler allows interruption
        if not self.handler.can_interrupt(step_name, self.state):
            return
        
        # Perform interruption
        self._perform_interrupt(step_name, checkpoint_data)
    
    def _perform_interrupt(self, step_name: str, checkpoint_data: Dict[str, Any]):
        """Execute the interrupt sequence"""
        # Update state
        self.state.set_status(PipelineStatus.PAUSED)
        self.state.clear_interrupt()
        
        # Create checkpoint
        checkpoint_info = self.handler.handle_interrupt(step_name, self.state, checkpoint_data)
        
        # Save checkpoint if callback is available
        if self.checkpoint_callback:
            self.checkpoint_callback(checkpoint_info)
        
        # Raise interrupt exception
        raise PipelineInterrupt(f"Pipeline paused at {step_name}", checkpoint_info)
    
    def request_pause(self):
        """Request pipeline pause at next interrupt point"""
        self.state.request_interrupt()
    
    def force_stop(self):
        """Force stop the pipeline immediately"""
        self.state.set_status(PipelineStatus.STOPPED)
        self.state.clear_interrupt()
    
    def can_pause(self) -> bool:
        """Check if pipeline can be paused"""
        return self.state.can_pause()
    
    def can_resume(self) -> bool:
        """Check if pipeline can be resumed"""
        return self.state.can_resume()


class InterventionManager:
    """
    Manages user interventions during pipeline execution
    Similar to C++ command pattern for user actions
    """
    
    def __init__(self):
        self.intervention_type: str = ""
        self.intervention_instruction: str = ""
        self.pending_intervention: bool = False
    
    def set_intervention(self, intervention_type: str, instruction: str):
        """Set user intervention details"""
        self.intervention_type = intervention_type
        self.intervention_instruction = instruction
        self.pending_intervention = True
    
    def get_intervention(self) -> Dict[str, str]:
        """Get and clear pending intervention"""
        if not self.pending_intervention:
            return {}
        
        intervention = {
            'type': self.intervention_type,
            'instruction': self.intervention_instruction
        }
        
        self.clear_intervention()
        return intervention
    
    def clear_intervention(self):
        """Clear current intervention"""
        self.intervention_type = ""
        self.intervention_instruction = ""
        self.pending_intervention = False
    
    def has_pending_intervention(self) -> bool:
        """Check if there's a pending intervention"""
        return self.pending_intervention 