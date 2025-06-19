"""
Main Entry Point for Quant Factor Pipeline UI
Similar to C++ main() function with proper module organization
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports (like C++ header includes)
from core import (
    PipelineState, PipelineStatus, InterruptController, 
    PipelineInterrupt, CheckpointManager, SessionManager,
    get_session_manager
)

# Component imports
from components.base_component import BaseComponent, StatefulComponent
from components.sidebar import SidebarComponent
from components.controls import ControlsComponent
from components.intervention import InterventionComponent
from components.results_display import ResultsDisplayComponent

# Pipeline imports
from pipeline.executor import PipelineExecutor


class QuanFactorPipelineApp:
    """
    Main application class
    Similar to C++ application class with dependency injection
    """
    
    def __init__(self):
        # Configure Streamlit page
        self._configure_page()
        
        # Initialize session manager (singleton pattern)
        self.session_manager = get_session_manager()
        self.session_manager.init_streamlit_session()
        
        # Initialize components (dependency injection pattern)
        self._initialize_components()
        
        # Initialize pipeline executor
        self.pipeline_executor = PipelineExecutor(self.session_manager)
    
    def _configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Quant Factor Pipeline",
            page_icon="📈", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS (like C++ resource definitions)
        self._load_custom_styles()
    
    def _load_custom_styles(self):
        """Load custom CSS styles"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 2rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .section-header {
                font-size: 1.2rem;
                font-weight: bold;
                color: #2e7d32;
                border-bottom: 2px solid #2e7d32;
                padding-bottom: 0.5rem;
                margin: 1rem 0;
            }
            .status-success {
                color: #2e7d32;
                font-weight: bold;
            }
            .status-error {
                color: #d32f2f;
                font-weight: bold;
            }
            .status-running {
                color: #f57c00;
                font-weight: bold;
            }
            .component-container {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def _initialize_components(self):
        """
        Initialize all UI components
        Similar to C++ constructor initialization lists
        """
        try:
            # Create components with dependency injection
            self.sidebar = SidebarComponent(self.session_manager)
            self.controls = ControlsComponent(self.session_manager)
            self.intervention = InterventionComponent(self.session_manager)
            self.results_display = ResultsDisplayComponent(self.session_manager)
            
            # Register component event handlers (observer pattern)
            self._register_event_handlers()
            
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            st.stop()
    
    def _register_event_handlers(self):
        """Register inter-component event handlers"""
        # Pipeline control events
        self.controls.register_event_handler('pause_requested', self._handle_pause_request)
        self.controls.register_event_handler('resume_requested', self._handle_resume_request)
        self.controls.register_event_handler('stop_requested', self._handle_stop_request)
        
        # Intervention events
        self.intervention.register_event_handler('intervention_submitted', self._handle_intervention)
        
        # Pipeline execution events
        self.sidebar.register_event_handler('pipeline_start_requested', self._handle_pipeline_start)
    
    def _handle_pause_request(self):
        """Handle pause request from controls"""
        self.session_manager.request_pause()
        st.info("暂停请求已提交，将在当前步骤完成后暂停")
    
    def _handle_resume_request(self):
        """Handle resume request from controls"""
        self.session_manager.sync_to_streamlit()
        st.info("管道执行已恢复")
        st.rerun()
    
    def _handle_stop_request(self):
        """Handle stop request from controls"""
        self.session_manager.force_stop()
        st.warning("管道执行已停止")
    
    def _handle_intervention(self, intervention_type: str, instruction: str):
        """Handle user intervention submission"""
        self.session_manager.set_intervention(intervention_type, instruction)
        st.success("用户干预已应用，管道将继续执行")
        st.rerun()
    
    def _handle_pipeline_start(self, human_input: str):
        """Handle pipeline start request"""
        if not human_input.strip():
            st.error("请先输入因子描述")
            return
        
        # Reset previous state
        self.session_manager.reset_session()
        
        # Start pipeline execution
        self.pipeline_executor.start_pipeline(human_input.strip())
    
    def run(self):
        """
        Main application run method
        Similar to C++ main application loop
        """
        try:
            # Update session state
            self.session_manager.sync_from_streamlit()
            
            # Render header
            self._render_header()
            
            # Render main layout
            self._render_layout()
            
            # Update display state
            self.session_manager.sync_to_streamlit()
            
        except Exception as e:
            st.error(f"应用运行错误: {str(e)}")
            
            # Show debug information in expander
            with st.expander("调试信息"):
                import traceback
                st.text(traceback.format_exc())
    
    def _render_header(self):
        """Render application header"""
        st.markdown('<div class="main-header">📈 量化因子管道仪表盘</div>', unsafe_allow_html=True)
    
    def _render_layout(self):
        """
        Render main application layout
        Similar to C++ UI layout management
        """
        # Sidebar with input and controls
        with st.sidebar:
            self.sidebar.render()
            st.markdown("---")
            self.controls.render()
        
        # Main content area
        main_container = st.container()
        
        with main_container:
            # User intervention panel (if paused)
            if self.session_manager.can_resume():
                self.intervention.render()
                st.markdown("---")
            
            # Results display
            self.results_display.render()
    
    def get_session_summary(self) -> dict:
        """
        Get comprehensive session summary
        Similar to C++ status/debug methods
        """
        return {
            'app_status': 'running',
            'session_manager': self.session_manager.get_state_summary(),
            'components': {
                'sidebar': self.sidebar.get_component_name(),
                'controls': self.controls.get_component_name(),
                'intervention': self.intervention.get_component_name(),
                'results_display': self.results_display.get_component_name()
            },
            'pipeline_executor': self.pipeline_executor.get_status() if hasattr(self.pipeline_executor, 'get_status') else 'unknown'
        }


def main():
    """
    Main entry point
    Similar to C++ main() function
    """
    try:
        # Create and run application
        app = QuanFactorPipelineApp()
        app.run()
        
    except KeyboardInterrupt:
        st.info("应用被用户中断")
    except Exception as e:
        st.error(f"应用启动失败: {str(e)}")
        
        # Show detailed error in development
        if os.getenv('STREAMLIT_ENV') == 'development':
            import traceback
            st.text(traceback.format_exc())


if __name__ == "__main__":
    main() 