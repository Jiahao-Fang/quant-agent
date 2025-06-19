"""
Tests for src/core/base_processor.py

Tests BaseProcessor functionality with new simplified architecture:
- Core abstract methods
- Capability detection
- Decorator integration
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState, ProcessorResult
from src.core.decorators import observable, evaluable, debuggable, interruptible


class TestProcessorState:
    """Test ProcessorState TypedDict structure."""
    
    def test_processor_state_structure(self):
        """Test ProcessorState structure and required fields."""
        state: ProcessorState = {
            'input_data': {"test": "data"},
            'output_data': None,
            'error': None,
            'status': 'idle',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        assert state['input_data'] == {"test": "data"}
        assert state['output_data'] is None
        assert state['error'] is None
        assert state['status'] == 'idle'
        assert state['retry_count'] == 0
        assert state['interrupt_requested'] is False
        assert state['save_point_id'] is None


class BasicProcessor(BaseProcessor):
    """Basic processor with no capabilities for testing."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """Simple processing logic."""
        state['output_data'] = {"processed": state.get('input_data')}
        state['status'] = 'success'
        return state


@observable(observers=["test_observer"])
class ObservableProcessor(BaseProcessor):
    """Processor with observable capability."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.FEATURE_BUILDER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"observable": True}
        state['status'] = 'success'
        return state


@evaluable(max_retries=2)
class EvaluableProcessor(BaseProcessor):
    """Processor with evaluable capability."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.FACTOR_AUGMENTER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"evaluable": True}
        state['status'] = 'success'
        return state
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """Evaluation logic."""
        state['eval_passed'] = True
        state['eval_reason'] = "Test evaluation passed"
        return state


@debuggable(max_retries=1)
class DebuggableProcessor(BaseProcessor):
    """Processor with debuggable capability."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.BACKTEST_RUNNER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"debuggable": True}
        state['status'] = 'success'
        return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """Debug logic."""
        state['should_retry'] = False
        state['debug_reason'] = "No errors to debug"
        return state


@interruptible(save_point_id="test_save_point")
class InterruptibleProcessor(BaseProcessor):
    """Processor with interruptible capability."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"interruptible": True}
        state['status'] = 'success'
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """Interrupt handling logic."""
        state['status'] = 'paused_by_user'
        return state


@observable(observers=["ui", "logger"])
@evaluable(max_retries=3)
@debuggable(max_retries=2)
@interruptible(save_point_id="full_processor")
class FullCapabilityProcessor(BaseProcessor):
    """Processor with all capabilities for testing."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"full_capabilities": True}
        state['status'] = 'success'
        return state
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        state['eval_passed'] = True
        state['eval_reason'] = "Full capability evaluation passed"
        return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        state['should_retry'] = True
        state['debug_reason'] = "Full capability debug"
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        state['status'] = 'paused_by_user'
        state['interrupt_data'] = {"paused": True}
        return state


class TestBaseProcessor:
    """Test BaseProcessor class functionality."""
    
    def test_basic_processor_initialization(self):
        """Test basic processor initialization."""
        config = {"test_param": "value"}
        processor = BasicProcessor(config)
        
        assert processor.config == config
        assert processor.get_processor_type() == ProcessorType.DATA_FETCHER
        assert processor.get_capabilities() == []
    
    def test_processor_with_capabilities(self):
        """Test processor capability detection."""
        observable_proc = ObservableProcessor({})
        evaluable_proc = EvaluableProcessor({})
        debuggable_proc = DebuggableProcessor({})
        interruptible_proc = InterruptibleProcessor({})
        full_proc = FullCapabilityProcessor({})
        
        assert observable_proc.get_capabilities() == ['observable']
        assert evaluable_proc.get_capabilities() == ['evaluable']
        assert debuggable_proc.get_capabilities() == ['debuggable']
        assert interruptible_proc.get_capabilities() == ['interruptible']
        
        full_capabilities = full_proc.get_capabilities()
        assert 'observable' in full_capabilities
        assert 'evaluable' in full_capabilities
        assert 'debuggable' in full_capabilities
        assert 'interruptible' in full_capabilities
    
    def test_capability_configuration(self):
        """Test capability configuration access."""
        evaluable_proc = EvaluableProcessor({})
        
        assert evaluable_proc.has_capability('evaluable')
        assert not evaluable_proc.has_capability('observable')
        
        eval_config = evaluable_proc.get_capability_config('evaluable')
        assert eval_config.get('max_retries') == 2
    
    def test_basic_processing(self):
        """Test basic processing functionality."""
        processor = BasicProcessor({})
        
        result = processor.process({"input": "test_data"})
        print(result)
        assert isinstance(result, ProcessorResult)
        assert result.success is True
        assert result.data == {"processed": {"input": "test_data"}}
        assert result.metadata["processor_type"] == "data_fetcher"
        assert result.metadata["capabilities"] == []
    
    @patch('src.core.base_processor.BaseProcessor.create_subgraph')
    def test_subgraph_creation(self, mock_create_subgraph):
        """Test subgraph creation."""
        mock_subgraph = Mock()
        mock_subgraph.invoke.return_value = {
            'output_data': {"test": "result"},
            'status': 'success',
            'retry_count': 0
        }
        mock_create_subgraph.return_value = mock_subgraph
        
        processor = BasicProcessor({})
        result = processor.process({"input": "test"})
       
        assert result.success is True
        assert result.data == {"test": "result"}
        assert result.metadata["retry_count"] == 0
        
        mock_create_subgraph.assert_called_once()
        mock_subgraph.invoke.assert_called_once()
        
        # Verify initial state structure
        call_args = mock_subgraph.invoke.call_args[0][0]
        assert call_args['input_data'] == {"input": "test"}
        assert call_args['status'] == 'initialized'
        assert call_args['save_point_id'] is None
    
    def test_processing_with_error(self):
        """Test processing with errors."""
        
        class FailingProcessor(BasicProcessor):
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                raise ValueError("Test error")
        
        processor = FailingProcessor({})
        result = processor.process({"input": "test"})
        
        assert result.success is False
        assert result.data is None
        assert isinstance(result.error, ValueError)
        assert "Test error" in str(result.error)
    
    def test_state_initialization(self):
        """Test processor state initialization."""
        processor = BasicProcessor({})
        input_data = {"test": "data"}
        
        state = processor._initialize_state(input_data)
        
        assert state['input_data'] == input_data
        assert state['output_data'] is None
        assert state['error'] is None
        assert state['status'] == 'initialized'
        assert state['retry_count'] == 0
        assert state['interrupt_requested'] is False
        assert state['save_point_id'] is None


class TestProcessorCapabilities:
    """Test capability-related functionality."""
    
    def test_capability_validation(self):
        """Test capability validation."""
        # Valid processors should initialize without error
        observable_proc = ObservableProcessor({})
        evaluable_proc = EvaluableProcessor({})
        debuggable_proc = DebuggableProcessor({})
        interruptible_proc = InterruptibleProcessor({})
        
        # Check capabilities are properly detected
        assert observable_proc.has_capability('observable')
        assert evaluable_proc.has_capability('evaluable')
        assert debuggable_proc.has_capability('debuggable')
        assert interruptible_proc.has_capability('interruptible')
    
    def test_missing_capability_methods(self):
        """Test validation of missing capability methods."""
        
        @evaluable(max_retries=1)
        class BadEvaluableProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            # Missing _evaluate_result method
        
        with pytest.raises(NotImplementedError):
            BadEvaluableProcessor({})
    
    def test_conditional_abstract_methods(self):
        """Test that conditional abstract methods work correctly."""
        # Processor without capabilities shouldn't require special methods
        basic_proc = BasicProcessor({})
        
        # Call the conditional methods - they should not raise NotImplementedError
        state = {'test': 'state'}
        result1 = basic_proc._evaluate_result(state)
        result2 = basic_proc._debug_error(state)
        result3 = basic_proc._handle_interrupt(state)
        
        # Should return state unchanged
        assert result1 == state
        assert result2 == state
        assert result3 == state
    
    def test_capability_config_access(self):
        """Test capability configuration access."""
        full_proc = FullCapabilityProcessor({})
        
        # Test evaluable config
        eval_config = full_proc.get_capability_config('evaluable')
        assert eval_config.get('max_retries') == 3
        
        # Test debuggable config
        debug_config = full_proc.get_capability_config('debuggable')
        assert debug_config.get('max_retries') == 2
        
        # Test interruptible config
        interrupt_config = full_proc.get_capability_config('interruptible')
        assert interrupt_config['save_point_id'] == "full_processor"
        
        # Test non-existent capability
        empty_config = full_proc.get_capability_config('non_existent')
        assert empty_config == {}


class TestProcessorIntegration:
    """Test processor integration scenarios."""
    
    def test_full_capability_processor(self):
        """Test processor with all capabilities."""
        processor = FullCapabilityProcessor({})
        
        capabilities = processor.get_capabilities()
        assert len(capabilities) == 4
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
        
        # Test capability configurations
        assert processor.get_capability_config('evaluable')['max_retries'] == 3
        assert processor.get_capability_config('debuggable')['max_retries'] == 2
        assert processor.get_capability_config('interruptible')['save_point_id'] == "full_processor"
    
    def test_processor_result_metadata(self):
        """Test processor result metadata."""
        processor = FullCapabilityProcessor({})
        
        result = processor.process({"test": "input"})
        
        assert result.metadata["processor_type"] == "data_fetcher"
        assert set(result.metadata["capabilities"]) == {'observable', 'evaluable', 'debuggable', 'interruptible'}
    
    def test_processor_type_validation(self):
        """Test processor type validation."""
        processors = [
            BasicProcessor({}),
            ObservableProcessor({}),
            EvaluableProcessor({}),
            DebuggableProcessor({}),
            InterruptibleProcessor({})
        ]
        
        expected_types = [
            ProcessorType.DATA_FETCHER,
            ProcessorType.FEATURE_BUILDER,
            ProcessorType.FACTOR_AUGMENTER,
            ProcessorType.BACKTEST_RUNNER,
            ProcessorType.DATA_FETCHER
        ]
        
        for processor, expected_type in zip(processors, expected_types):
            assert processor.get_processor_type() == expected_type


class TestErrorScenarios:
    """Test error handling scenarios."""
    
    def test_invalid_config(self):
        """Test processors with invalid configurations."""
        # Should still work with invalid config
        processor = BasicProcessor({"invalid": "config"})
        assert processor.config == {"invalid": "config"}
        assert processor.get_capabilities() == []
    
    def test_empty_input_processing(self):
        """Test processing with empty input."""
        processor = BasicProcessor({})
        result = processor.process({})
        
        assert result.success is True
        assert result.data == {"processed": {}}
    
    def test_none_input_processing(self):
        """Test processing with None input."""
        processor = BasicProcessor({})
        result = processor.process(None)
        
        assert result.success is True
        assert result.data == {"processed": None}
    
    @patch('src.core.base_processor.BaseProcessor.create_subgraph')
    def test_subgraph_execution_error(self, mock_create_subgraph):
        """Test subgraph execution error handling."""
        mock_subgraph = Mock()
        mock_subgraph.invoke.side_effect = RuntimeError("Subgraph execution failed")
        mock_create_subgraph.return_value = mock_subgraph
        
        processor = BasicProcessor({})
        result = processor.process({"input": "test"})
        
        assert result.success is False
        assert result.data is None
        assert isinstance(result.error, RuntimeError)
        assert "Subgraph execution failed" in str(result.error) 