"""
Tests for src/core/pipeline_coordinator.py

Tests PipelineCoordinator functionality:
- LangGraph workflow creation
- Processor management
- Interrupt handling
- Pipeline execution
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.core.pipeline_coordinator import PipelineCoordinator, PipelineConfig, PipelineResult
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from src.core.decorators import observable, interruptible


class BasicTestProcessor(BaseProcessor):
    """Basic processor for testing."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"basic": "result"}
        state['status'] = 'success'
        return state


@observable(observers=["test_observer"])
class ObservableTestProcessor(BaseProcessor):
    """Observable processor for testing."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.FEATURE_BUILDER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"observable": "result"}
        state['status'] = 'success'
        return state


@interruptible(save_point_id="test_save_point")
class InterruptibleTestProcessor(BaseProcessor):
    """Interruptible processor for testing."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.BACKTEST_RUNNER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"interruptible": "result"}
        state['status'] = 'success'
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        state['status'] = 'paused'
        return state


class TestPipelineCoordinator:
    """Test PipelineCoordinator functionality."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        coordinator = PipelineCoordinator()
        
        assert hasattr(coordinator, 'processors')
        assert hasattr(coordinator, 'workflow')
        assert coordinator.workflow is None
    
    def test_add_processor(self):
        """Test adding processors to coordinator."""
        coordinator = PipelineCoordinator()
        processor = BasicTestProcessor({})
        
        coordinator.add_processor("data_fetcher", processor)
        
        assert "data_fetcher" in coordinator.processors
        assert coordinator.processors["data_fetcher"] == processor
        
    def test_remove_processor(self):
        """Test removing processors from coordinator."""
        coordinator = PipelineCoordinator()
        processor = BasicTestProcessor({})
        
        coordinator.add_processor("data_fetcher", processor)
        coordinator.remove_processor("data_fetcher")
        
        assert "data_fetcher" not in coordinator.processors
        
    def test_get_processor(self):
        """Test getting processor by name."""
        coordinator = PipelineCoordinator()
        processor = BasicTestProcessor({})
        
        coordinator.add_processor("data_fetcher", processor)
        
        retrieved = coordinator.get_processor("data_fetcher")
        assert retrieved == processor
        
        # Test non-existent processor
        assert coordinator.get_processor("nonexistent") is None


class TestWorkflowCreation:
    """Test LangGraph workflow creation."""
    
    @patch('src.core.pipeline_coordinator.StateGraph')
    def test_create_workflow_basic(self, mock_state_graph):
        """Test basic workflow creation."""
        coordinator = PipelineCoordinator()
        
        # Add processors
        coordinator.add_processor("data_fetcher", BasicTestProcessor({}))
        coordinator.add_processor("feature_builder", ObservableTestProcessor({}))
        
        # Create workflow
        coordinator.create_workflow()
        
        assert coordinator.workflow is not None
        mock_state_graph.assert_called_once()
    
    @patch('src.core.pipeline_coordinator.StateGraph')
    def test_create_workflow_with_interrupts(self, mock_state_graph):
        """Test workflow creation with interruptible processors."""
        coordinator = PipelineCoordinator()
        
        # Add processors with interrupt capability
        coordinator.add_processor("data_fetcher", BasicTestProcessor({}))
        coordinator.add_processor("backtest", InterruptibleTestProcessor({}))
        
        # Create workflow
        coordinator.create_workflow()
        
        assert coordinator.workflow is not None
        mock_state_graph.assert_called_once()
        
    def test_workflow_creation_empty_processors(self):
        """Test workflow creation with no processors."""
        coordinator = PipelineCoordinator()
        
        # Should handle empty processors gracefully
        coordinator.create_workflow()
        
        assert coordinator.workflow is not None


class TestPipelineExecution:
    """Test pipeline execution functionality."""
    
    @patch('src.core.pipeline_coordinator.PipelineCoordinator.create_workflow')
    def test_execute_pipeline_basic(self, mock_create_workflow):
        """Test basic pipeline execution."""
        coordinator = PipelineCoordinator()
        
        # Setup mock workflow
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {
            'data_fetcher_output': {"fetched": "data"},
            'feature_builder_output': {"features": "built"},
            'status': 'success'
        }
        coordinator.workflow = mock_workflow
        
        # Add processors
        coordinator.add_processor("data_fetcher", BasicTestProcessor({}))
        coordinator.add_processor("feature_builder", ObservableTestProcessor({}))
        
        # Execute pipeline
        result = coordinator.execute_pipeline({"input": "test_data"})
        
        assert result is not None
        mock_workflow.invoke.assert_called_once()
    
    def test_execute_pipeline_no_workflow(self):
        """Test pipeline execution with no workflow created."""
        coordinator = PipelineCoordinator()
        
        # Add processors but don't create workflow
        coordinator.add_processor("data_fetcher", BasicTestProcessor({}))
        
        # Execute should create workflow automatically
        result = coordinator.execute_pipeline({"input": "test_data"})
        
        assert result is not None
        assert coordinator.workflow is not None


class TestInterruptHandling:
    """Test interrupt handling in pipeline."""
    
    def test_interrupt_pipeline(self):
        """Test interrupting a pipeline with interruptible processors."""
        coordinator = PipelineCoordinator()
        
        # Add interruptible processor
        coordinator.add_processor("backtest", InterruptibleTestProcessor({}))
        
        # Mock workflow with interrupt support
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {'status': 'interrupted'}
        coordinator.workflow = mock_workflow
        
        # Execute with interrupt
        result = coordinator.execute_pipeline({"input": "test", "interrupt": True})
        
        assert result is not None
        mock_workflow.invoke.assert_called_once()
    
    def test_resume_pipeline(self):
        """Test resuming a paused pipeline."""
        coordinator = PipelineCoordinator()
        
        # Add processors
        coordinator.add_processor("backtest", InterruptibleTestProcessor({}))
        
        # Mock workflow resume
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {'status': 'resumed'}
        coordinator.workflow = mock_workflow
        
        # Resume execution
        result = coordinator.resume_pipeline({"save_point_id": "test_save_point"})
        
        assert result is not None
        mock_workflow.invoke.assert_called_once()


class TestErrorHandling:
    """Test error handling in pipeline coordinator."""
    
    def test_processor_not_found_error(self):
        """Test handling of processor not found error."""
        coordinator = PipelineCoordinator()
        
        # Try to get non-existent processor
        result = coordinator.get_processor("nonexistent")
        
        assert result is None
    
    def test_remove_nonexistent_processor(self):
        """Test removing non-existent processor."""
        coordinator = PipelineCoordinator()
        
        # Should handle gracefully
        coordinator.remove_processor("nonexistent")
        
        # No exception should be raised
        assert True
    
    def test_workflow_execution_error(self):
        """Test workflow execution error handling."""
        coordinator = PipelineCoordinator()
        
        # Setup mock workflow that raises error
        mock_workflow = Mock()
        mock_workflow.invoke.side_effect = RuntimeError("Workflow execution failed")
        coordinator.workflow = mock_workflow
        
        # Execute should handle error gracefully
        with pytest.raises(RuntimeError):
            coordinator.execute_pipeline({"input": "test_data"})


class TestCapabilityDetection:
    """Test capability detection and workflow adaptation."""
    
    def test_detect_interruptible_processors(self):
        """Test detection of interruptible processors."""
        coordinator = PipelineCoordinator()
        
        # Add mix of processors
        coordinator.add_processor("basic", BasicTestProcessor({}))
        coordinator.add_processor("interruptible", InterruptibleTestProcessor({}))
        
        # Check capability detection
        basic_proc = coordinator.get_processor("basic")
        interrupt_proc = coordinator.get_processor("interruptible")
        
        assert not basic_proc.has_capability('interruptible')
        assert interrupt_proc.has_capability('interruptible')
    
    def test_capability_based_workflow_creation(self):
        """Test workflow creation adapts to processor capabilities."""
        coordinator = PipelineCoordinator()
        
        # Add processors with different capabilities
        coordinator.add_processor("basic", BasicTestProcessor({}))
        coordinator.add_processor("observable", ObservableTestProcessor({}))
        coordinator.add_processor("interruptible", InterruptibleTestProcessor({}))
        
        # Create workflow
        coordinator.create_workflow()
        
        # Workflow should be created successfully
        assert coordinator.workflow is not None


class TestPipelineIntegration:
    """Test full pipeline integration scenarios."""
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution flow."""
        coordinator = PipelineCoordinator()
        
        # Setup processors
        coordinator.add_processor("data_fetcher", BasicTestProcessor({}))
        coordinator.add_processor("feature_builder", ObservableTestProcessor({}))
        
        # Mock successful execution
        with patch.object(coordinator, 'create_workflow') as mock_create:
            mock_workflow = Mock()
            mock_workflow.invoke.return_value = {
                'output_data': {"final": "result"},
                'status': 'success'
            }
            coordinator.workflow = mock_workflow
            
            result = coordinator.execute_pipeline({"input": "test_data"})
            
            assert result is not None
            mock_workflow.invoke.assert_called_once()
    
    def test_pipeline_with_all_capabilities(self):
        """Test pipeline with processors having all capabilities."""
        coordinator = PipelineCoordinator()
        
        # Add processors with various capabilities
        coordinator.add_processor("observable", ObservableTestProcessor({}))
        coordinator.add_processor("interruptible", InterruptibleTestProcessor({}))
        
        # Create and execute
        coordinator.create_workflow()
        
        # Mock execution
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {'status': 'success'}
        coordinator.workflow = mock_workflow
        
        result = coordinator.execute_pipeline({"input": "test_data"})
        
        assert result is not None 