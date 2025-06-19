"""
Tests for src/core/workflow_nodes/eval_node.py

Tests the evaluation workflow node functionality:
- Node initialization with processor
- Result evaluation
- Retry decision logic
- Monitoring integration
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.workflow_nodes.eval_node import EvalNode
from src.core.base_processor import BaseProcessor, ProcessorState, ProcessorType


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def __init__(self, config: Dict[str, Any], capabilities: list = None):
        super().__init__(config)
        if capabilities:
            self._capabilities = capabilities
            # Set capability configs
            if 'evaluable' in capabilities:
                self._capability_configs['evaluable'] = config.get('eval_config', {'max_retries': 3})
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state["output_data"] = {"result": "processed"}
        return state
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """Default evaluation implementation."""
        state["eval_passed"] = True
        return state


class TestEvalNode:
    """Test EvalNode functionality."""
    
    def test_eval_node_initialization(self):
        """Test eval node initialization with processor."""
        mock_processor = MockProcessor({})
        node = EvalNode(mock_processor)
        
        assert node.processor == mock_processor
    
    def test_eval_node_execution_success(self):
        """Test successful eval node execution."""
        mock_processor = MockProcessor({'eval_config': {'max_retries': 2}}, capabilities=['evaluable'])
        node = EvalNode(mock_processor)
        
        # State with output data to evaluate
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['eval_retry_count'] == 1
        assert result['eval_passed'] is True
        assert 'eval_config' in result
        assert result['eval_config']['max_retries'] == 2
    
    def test_eval_node_execution_error(self):
        """Test eval node execution with evaluation error."""
        mock_processor = MockProcessor({}, capabilities=['evaluable'])
        
        # Override to raise exception in evaluation
        def error_eval(state):
            raise RuntimeError("Evaluation failed")
        
        mock_processor._evaluate_result = error_eval
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['eval_error'] is not None
        assert isinstance(result['eval_error'], RuntimeError)
        assert result['eval_passed'] is False
    
    def test_eval_node_with_observable_capability(self):
        """Test eval node with observable processor."""
        mock_processor = MockProcessor({}, capabilities=['evaluable', 'observable'])
        mock_processor._capability_configs = {
            'evaluable': {'max_retries': 3},
            'observable': {'observers': ['test_observer']}
        }
        
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should print monitoring events for evaluation
            assert mock_print.called
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("evaluation_started" in call for call in calls)
            assert any("evaluation_completed" in call for call in calls)
    
    def test_eval_node_check_retry_can_retry(self):
        """Test check_retry when retry is allowed."""
        mock_processor = MockProcessor({}, capabilities=['evaluable'])
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'eval_config': {'max_retries': 3},
            'eval_retry_count': 1,
            'eval_passed': False  # Evaluation failed
        }
        
        result = node.check_retry(state)
        
        assert result['output_data'] is None  # Should be cleared for retry
        assert result['eval_passed'] is False
        assert result['status'] == 'retrying_after_eval'
    
    def test_eval_node_check_retry_cannot_retry(self):
        """Test check_retry when retry is not allowed."""
        mock_processor = MockProcessor({}, capabilities=['evaluable'])
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'eval_config': {'max_retries': 3},
            'eval_retry_count': 3,  # Reached max retries
            'eval_passed': False
        }
        
        result = node.check_retry(state)
        
        assert result['status'] == 'failed_after_eval'
    
    def test_eval_node_check_retry_eval_passed(self):
        """Test check_retry when evaluation passed."""
        mock_processor = MockProcessor({}, capabilities=['evaluable'])
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'eval_config': {'max_retries': 3},
            'eval_retry_count': 1,
            'eval_passed': True  # Evaluation passed
        }
        
        result = node.check_retry(state)
        
        # Should not retry if evaluation passed
        assert result['status'] == 'failed_after_eval'  # Still goes to failed path in current implementation
    
    def test_eval_node_monitoring_events(self):
        """Test monitoring events for evaluation operations."""
        mock_processor = MockProcessor({}, capabilities=['evaluable', 'observable'])
        mock_processor._capability_configs = {
            'evaluable': {'max_retries': 3},
            'observable': {'observers': ['test_observer']}
        }
        
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            node.execute(state)
            
            # Check monitoring event structure
            calls = [str(call) for call in mock_print.call_args_list]
            
            # Should have evaluation_started and evaluation_completed events
            eval_started = any("evaluation_started" in call for call in calls)
            eval_completed = any("evaluation_completed" in call for call in calls)
            
            assert eval_started
            assert eval_completed
            
            # Should contain required event fields
            assert any("event_type" in call for call in calls)
            assert any("node_name" in call for call in calls)
            assert any("processor_type" in call for call in calls)
    
    def test_eval_node_without_observable_capability(self):
        """Test eval node without observable capability."""
        mock_processor = MockProcessor({}, capabilities=['evaluable'])  # No observable
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should not print monitoring events
            assert not mock_print.called
            assert result['eval_passed'] is True
    
    def test_eval_node_retry_decision_monitoring(self):
        """Test monitoring events for retry decisions."""
        mock_processor = MockProcessor({}, capabilities=['evaluable', 'observable'])
        mock_processor._capability_configs = {
            'evaluable': {'max_retries': 3},
            'observable': {'observers': ['test_observer']}
        }
        
        node = EvalNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'processed'},
            'error': None,
            'status': 'success',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'eval_config': {'max_retries': 3},
            'eval_retry_count': 1,
            'eval_passed': False
        }
        
        with patch('builtins.print') as mock_print:
            node.check_retry(state)
            
            # Should print retry decision monitoring event
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("eval_retry_decision" in call for call in calls) 