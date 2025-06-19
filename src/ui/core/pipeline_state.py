"""
Pipeline State Management Module
Equivalent to C++ class header for state management
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import time


class PipelineStatus(Enum):
    """Pipeline execution status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class StepResult:
    """Data class for storing step execution results"""
    step_name: str
    timestamp: float
    status: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'step_name': self.step_name,
            'timestamp': self.timestamp,
            'status': self.status,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepResult':
        """Create from dictionary"""
        return cls(
            step_name=data['step_name'],
            timestamp=data['timestamp'],
            status=data['status'],
            data=data['data']
        )


class PipelineState:
    """
    Central state management class
    Similar to C++ singleton pattern for global state
    """
    
    def __init__(self):
        # Execution state
        self.status: PipelineStatus = PipelineStatus.IDLE
        self.current_step: str = ""
        self.progress: float = 0.0
        
        # Pipeline data
        self.feature_description: str = ""
        self.query_info: str = ""
        self.data: Dict[str, Any] = {}
        self.feature_table: Optional[Any] = None
        
        # History and tracking
        self.step_results_history: List[StepResult] = []
        self.code_history: List[Dict[str, Any]] = []
        self.debug_history: List[str] = []
        
        # Control state
        self.interrupt_requested: bool = False
        self.max_retries: int = 3
        self.current_retry: int = 0
        
    def add_step_result(self, step_name: str, result_data: Dict[str, Any], status: str = "completed"):
        """Add a step result to the history"""
        step_result = StepResult(
            step_name=step_name,
            timestamp=time.time(),
            status=status,
            data=result_data
        )
        self.step_results_history.append(step_result)
    
    def update_progress(self, progress: float):
        """Update pipeline progress (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
    
    def set_status(self, status: PipelineStatus):
        """Update pipeline status"""
        self.status = status
    
    def is_running(self) -> bool:
        """Check if pipeline is currently running"""
        return self.status == PipelineStatus.RUNNING
    
    def is_paused(self) -> bool:
        """Check if pipeline is paused"""
        return self.status == PipelineStatus.PAUSED
    
    def can_pause(self) -> bool:
        """Check if pipeline can be paused"""
        return self.status == PipelineStatus.RUNNING
    
    def can_resume(self) -> bool:
        """Check if pipeline can be resumed"""
        return self.status == PipelineStatus.PAUSED
    
    def request_interrupt(self):
        """Request pipeline interruption"""
        if self.can_pause():
            self.interrupt_requested = True
    
    def clear_interrupt(self):
        """Clear interrupt request"""
        self.interrupt_requested = False
    
    def reset(self):
        """Reset all state to initial values"""
        self.__init__()
    
    def get_latest_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get the latest result for a specific step"""
        for result in reversed(self.step_results_history):
            if result.step_name == step_name:
                return result
        return None
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for serialization"""
        return {
            'status': self.status.value,
            'current_step': self.current_step,
            'progress': self.progress,
            'feature_description': self.feature_description,
            'query_info': self.query_info,
            'data_keys': list(self.data.keys()) if self.data else [],
            'step_results_history': [result.to_dict() for result in self.step_results_history],
            'code_history': self.code_history,
            'debug_history': self.debug_history,
            'current_retry': self.current_retry,
            'max_retries': self.max_retries
        }
    
    def import_state(self, state_data: Dict[str, Any]):
        """Import state from serialized data"""
        self.status = PipelineStatus(state_data.get('status', 'idle'))
        self.current_step = state_data.get('current_step', '')
        self.progress = state_data.get('progress', 0.0)
        self.feature_description = state_data.get('feature_description', '')
        self.query_info = state_data.get('query_info', '')
        self.code_history = state_data.get('code_history', [])
        self.debug_history = state_data.get('debug_history', [])
        self.current_retry = state_data.get('current_retry', 0)
        self.max_retries = state_data.get('max_retries', 3)
        
        # Restore step results
        self.step_results_history = [
            StepResult.from_dict(result_data) 
            for result_data in state_data.get('step_results_history', [])
        ] 