"""
Tests for src/core/workflow_nodes/debug_node.py

Tests the debug workflow node functionality:
- Node initialization with processor
- Debug execution for errors
- Retry decision logic
- Monitoring integration
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.workflow_nodes.debug_node import DebugNode
from src.core.base_processor import BaseProcessor, ProcessorState, ProcessorType


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def __init__(self, config: Dict[str, Any], capabilities: list = None):
        super().__init__(config)
        if capabilities:
            self._capabilities = capabilities
            # Set capability configs
            if 'debuggable' in capabilities:
                self._capability_configs['debuggable'] = config.get('debug_config', {'max_retries': 3})
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state["output_data"] = {"result": "processed"}
        return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """Default debug implementation."""
        state["should_retry"] = True
        return state


class TestDebugNode:
    """Test DebugNode functionality."""
    
    def test_debug_node_initialization(self):
        """Test debug node initialization with processor."""
        mock_processor = MockProcessor({})
        node = DebugNode(mock_processor)
        
        assert node.processor == mock_processor
    
    def test_debug_node_execution_success(self):
        """Test successful debug node execution."""
        mock_processor = MockProcessor({'debug_config': {'max_retries': 2}}, capabilities=['debuggable'])
        node = DebugNode(mock_processor)
        
        # State with error to debug
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['debug_retry_count'] == 1
        assert result['should_retry'] is True
        assert 'debug_config' in result
        assert result['debug_config']['max_retries'] == 2
    
    def test_debug_node_execution_error(self):
        """Test debug node execution with debug logic error."""
        mock_processor = MockProcessor({}, capabilities=['debuggable'])
        
        # Override to raise exception in debug
        def error_debug(state):
            raise RuntimeError("Debug failed")
        
        mock_processor._debug_error = error_debug
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['debug_error'] is not None
        assert isinstance(result['debug_error'], RuntimeError)
        assert result['should_retry'] is False
    
    def test_debug_node_with_observable_capability(self):
        """Test debug node with observable processor."""
        mock_processor = MockProcessor({}, capabilities=['debuggable', 'observable'])
        mock_processor._capability_configs = {
            'debuggable': {'max_retries': 3},
            'observable': {'observers': ['test_observer']}
        }
        
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should print monitoring events for debug
            assert mock_print.called
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("debug_started" in call for call in calls)
            assert any("debug_completed" in call for call in calls)
    
    def test_debug_node_check_retry_can_retry(self):
        """Test check_retry when retry is allowed."""
        mock_processor = MockProcessor({}, capabilities=['debuggable'])
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'debug_config': {'max_retries': 3},
            'debug_retry_count': 1,
            'should_retry': True
        }
        
        result = node.check_retry(state)
        
        assert result['error'] is None  # Should be cleared for retry
        assert result['status'] == 'retrying_after_debug'
    
    def test_debug_node_check_retry_cannot_retry(self):
        """Test check_retry when retry is not allowed."""
        mock_processor = MockProcessor({}, capabilities=['debuggable'])
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'debug_config': {'max_retries': 3},
            'debug_retry_count': 3,  # Reached max retries
            'should_retry': True
        }
        
        result = node.check_retry(state)
        
        assert result['status'] == 'failed_after_debug'
    
    def test_debug_node_check_retry_should_not_retry(self):
        """Test check_retry when debug logic says not to retry."""
        mock_processor = MockProcessor({}, capabilities=['debuggable'])
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None,
            'debug_config': {'max_retries': 3},
            'debug_retry_count': 1,
            'should_retry': False  # Debug logic says don't retry
        }
        
        result = node.check_retry(state)
        
        assert result['status'] == 'failed_after_debug'
    
    def test_debug_node_monitoring_events(self):
        """Test monitoring events for debug operations."""
        mock_processor = MockProcessor({}, capabilities=['debuggable', 'observable'])
        mock_processor._capability_configs = {
            'debuggable': {'max_retries': 3},
            'observable': {'observers': ['test_observer']}
        }
        
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            node.execute(state)
            
            # Check monitoring event structure
            calls = [str(call) for call in mock_print.call_args_list]
            
            # Should have debug_started and debug_completed events
            debug_started = any("debug_started" in call for call in calls)
            debug_completed = any("debug_completed" in call for call in calls)
            
            assert debug_started
            assert debug_completed
            
            # Should contain required event fields
            assert any("event_type" in call for call in calls)
            assert any("node_name" in call for call in calls)
            assert any("processor_type" in call for call in calls)
    
    def test_debug_node_without_observable_capability(self):
        """Test debug node without observable capability."""
        mock_processor = MockProcessor({}, capabilities=['debuggable'])  # No observable
        node = DebugNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Processing failed"),
            'status': 'error',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should not print monitoring events
            assert not mock_print.called
            assert result['should_retry'] is True 