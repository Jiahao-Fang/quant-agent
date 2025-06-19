"""
Base Component Module
Similar to C++ pure virtual base classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import streamlit as st
from ..core.session_manager import SessionManager


class BaseComponent(ABC):
    """
    Abstract base class for all UI components
    Similar to C++ pure virtual base class with interface definition
    """
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.pipeline_state = session_manager.get_pipeline_state()
        self.interrupt_controller = session_manager.get_interrupt_controller()
        self.intervention_manager = session_manager.get_intervention_manager()
        self.checkpoint_manager = session_manager.get_checkpoint_manager()
    
    @abstractmethod
    def render(self) -> None:
        """
        Render the component
        Pure virtual method that must be implemented by derived classes
        """
        pass
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get the component name for identification"""
        pass
    
    def update_state(self) -> None:
        """
        Update component state (optional override)
        Similar to C++ virtual methods with default implementation
        """
        # Sync from Streamlit to internal state
        self.session_manager.sync_from_streamlit()
    
    def refresh_display(self) -> None:
        """
        Refresh the display (optional override)
        Similar to C++ virtual methods with default implementation
        """
        # Sync internal state to Streamlit
        self.session_manager.sync_to_streamlit()
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle component errors
        Similar to C++ exception handling patterns
        """
        error_msg = f"Error in {self.get_component_name()}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        
        st.error(error_msg)
        
        # Log to session state for debugging
        if 'component_errors' not in st.session_state:
            st.session_state.component_errors = []
        
        st.session_state.component_errors.append({
            'component': self.get_component_name(),
            'context': context,
            'error': str(error),
            'timestamp': st.session_state.get('last_update_time', 0)
        })
    
    def is_enabled(self) -> bool:
        """
        Check if component should be enabled
        Virtual method for conditional rendering
        """
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get component configuration
        Virtual method for component settings
        """
        return {}


class StatefulComponent(BaseComponent):
    """
    Base class for components that maintain state
    Similar to C++ template classes with state management
    """
    
    def __init__(self, session_manager: SessionManager, state_key: str):
        super().__init__(session_manager)
        self.state_key = state_key
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize component-specific state in Streamlit session"""
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = self._get_default_state()
    
    def _get_default_state(self) -> Dict[str, Any]:
        """
        Get default state for the component
        Virtual method to be overridden by derived classes
        """
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current component state"""
        return st.session_state.get(self.state_key, {})
    
    def update_component_state(self, updates: Dict[str, Any]):
        """Update component state"""
        current_state = self.get_state()
        current_state.update(updates)
        st.session_state[self.state_key] = current_state
    
    def reset_component_state(self):
        """Reset component state to default"""
        st.session_state[self.state_key] = self._get_default_state()


class InteractiveComponent(StatefulComponent):
    """
    Base class for interactive components with event handling
    Similar to C++ event-driven programming patterns
    """
    
    def __init__(self, session_manager: SessionManager, state_key: str):
        super().__init__(session_manager, state_key)
        self.event_handlers = {}
    
    def register_event_handler(self, event_name: str, handler_func: callable):
        """
        Register an event handler
        Similar to C++ callback registration
        """
        self.event_handlers[event_name] = handler_func
    
    def trigger_event(self, event_name: str, *args, **kwargs):
        """
        Trigger an event if handler is registered
        Similar to C++ event dispatching
        """
        handler = self.event_handlers.get(event_name)
        if handler:
            try:
                return handler(*args, **kwargs)
            except Exception as e:
                self.handle_error(e, f"Event handler '{event_name}'")
                return None
        return None
    
    def handle_user_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Handle user input (virtual method)
        Returns True if input was handled successfully
        """
        return True 