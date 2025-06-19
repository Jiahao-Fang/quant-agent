"""
Session Manager Module
Similar to C++ singleton pattern and resource management
"""

import streamlit as st
from typing import Dict, Any, Optional
from .pipeline_state import PipelineState, PipelineStatus
from .interrupt_system import InterruptController, InterventionManager
from .checkpoint import CheckpointManager


class SessionManager:
    """
    Centralized session state management
    Similar to C++ singleton pattern with thread safety considerations
    """
    
    _instance: Optional['SessionManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'SessionManager':
        """Ensure singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize session manager (only once)"""
        if SessionManager._initialized:
            return
        
        # Core components (like C++ member initialization)
        self.pipeline_state = PipelineState()
        self.interrupt_controller = InterruptController(self.pipeline_state)
        self.intervention_manager = InterventionManager()
        self.checkpoint_manager = CheckpointManager()
        
        # UI state keys (like C++ static members)
        self.UI_STATE_KEYS = {
            'pipeline_running': 'pipeline_running',
            'pipeline_paused': 'pipeline_paused', 
            'user_intervention': 'user_intervention',
            'intervention_type': 'intervention_type',
            'current_checkpoint': 'current_checkpoint'
        }
        
        SessionManager._initialized = True
    
    def init_streamlit_session(self):
        """
        Initialize Streamlit session state
        Similar to C++ constructor initialization lists
        """
        # Pipeline execution state
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
        if 'pipeline_paused' not in st.session_state:
            st.session_state.pipeline_paused = False
        
        # User intervention state
        if 'user_intervention' not in st.session_state:
            st.session_state.user_intervention = ""
        if 'intervention_type' not in st.session_state:
            st.session_state.intervention_type = ""
        
        # Checkpoint state
        if 'current_checkpoint' not in st.session_state:
            st.session_state.current_checkpoint = None
        
        # Pipeline data (preserved across runs)
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = None
        if 'code_history' not in st.session_state:
            st.session_state.code_history = []
        if 'query_info' not in st.session_state:
            st.session_state.query_info = ""
        if 'step_results_history' not in st.session_state:
            st.session_state.step_results_history = []
    
    def sync_from_streamlit(self):
        """
        Sync state from Streamlit session to internal state
        Similar to C++ data synchronization patterns
        """
        # Update pipeline status based on Streamlit state
        if st.session_state.get('pipeline_running', False):
            if st.session_state.get('pipeline_paused', False):
                self.pipeline_state.set_status(PipelineStatus.PAUSED)
            else:
                self.pipeline_state.set_status(PipelineStatus.RUNNING)
        else:
            if self.pipeline_state.status == PipelineStatus.RUNNING:
                self.pipeline_state.set_status(PipelineStatus.IDLE)
        
        # Sync intervention state
        if st.session_state.get('user_intervention'):
            self.intervention_manager.set_intervention(
                st.session_state.get('intervention_type', ''),
                st.session_state.get('user_intervention', '')
            )
        
        # Sync checkpoint data
        if st.session_state.get('current_checkpoint'):
            # This would need checkpoint deserialization
            pass
    
    def sync_to_streamlit(self):
        """
        Sync internal state to Streamlit session
        Similar to C++ observer pattern notifications
        """
        # Update Streamlit state from pipeline state
        st.session_state.pipeline_running = self.pipeline_state.is_running()
        st.session_state.pipeline_paused = self.pipeline_state.is_paused()
        
        # Update intervention state
        if self.intervention_manager.has_pending_intervention():
            intervention = self.intervention_manager.get_intervention()
            st.session_state.intervention_type = intervention.get('type', '')
            st.session_state.user_intervention = intervention.get('instruction', '')
        
        # Update checkpoint state
        current_checkpoint = self.checkpoint_manager.get_current_checkpoint()
        if current_checkpoint:
            st.session_state.current_checkpoint = current_checkpoint.to_dict()
        else:
            st.session_state.current_checkpoint = None
    
    def get_pipeline_state(self) -> PipelineState:
        """Get the pipeline state instance"""
        return self.pipeline_state
    
    def get_interrupt_controller(self) -> InterruptController:
        """Get the interrupt controller instance"""
        return self.interrupt_controller
    
    def get_intervention_manager(self) -> InterventionManager:
        """Get the intervention manager instance"""
        return self.intervention_manager
    
    def get_checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager instance"""
        return self.checkpoint_manager
    
    def reset_session(self):
        """
        Reset all session data
        Similar to C++ destructor cleanup
        """
        # Reset internal state
        self.pipeline_state.reset()
        self.intervention_manager.clear_intervention()
        self.checkpoint_manager.clear_current_checkpoint()
        
        # Reset Streamlit session state
        for key in self.UI_STATE_KEYS.values():
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset data keys
        data_keys = [
            'pipeline_results', 'code_history', 'query_info', 
            'step_results_history', 'current_step', 'step_status'
        ]
        for key in data_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Re-initialize
        self.init_streamlit_session()
    
    def create_checkpoint(self, step_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a checkpoint with current state"""
        return self.checkpoint_manager.create_checkpoint(
            step_name=step_name,
            state=self.pipeline_state,
            metadata=metadata
        )
    
    def request_pause(self):
        """Request pipeline pause"""
        self.interrupt_controller.request_pause()
        self.sync_to_streamlit()
    
    def force_stop(self):
        """Force stop the pipeline"""
        self.interrupt_controller.force_stop()
        self.sync_to_streamlit()
    
    def can_pause(self) -> bool:
        """Check if pipeline can be paused"""
        return self.interrupt_controller.can_pause()
    
    def can_resume(self) -> bool:
        """Check if pipeline can be resumed"""
        return self.interrupt_controller.can_resume()
    
    def set_intervention(self, intervention_type: str, instruction: str):
        """Set user intervention"""
        self.intervention_manager.set_intervention(intervention_type, instruction)
        self.sync_to_streamlit()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive state summary
        Similar to C++ debug/status reporting
        """
        return {
            'pipeline_status': self.pipeline_state.status.value,
            'current_step': self.pipeline_state.current_step,
            'progress': self.pipeline_state.progress,
            'can_pause': self.can_pause(),
            'can_resume': self.can_resume(),
            'has_intervention': self.intervention_manager.has_pending_intervention(),
            'checkpoint_count': len(self.checkpoint_manager.list_checkpoints()),
            'current_checkpoint': self.checkpoint_manager.current_checkpoint is not None,
            'step_results_count': len(self.pipeline_state.step_results_history)
        }


# Global instance accessor (like C++ extern declarations)
def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    return SessionManager() 