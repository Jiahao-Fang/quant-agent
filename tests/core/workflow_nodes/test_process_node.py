"""
Tests for src/core/workflow_nodes/process_node.py

Tests the core processing node functionality:
- Node initialization with processor
- Successful execution
- Error handling
- Monitoring integration
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.workflow_nodes.process_node import ProcessNode
from src.core.base_processor import BaseProcessor, ProcessorState, ProcessorType


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def __init__(self, config: Dict[str, Any], capabilities: list = None):
        super().__init__(config)
        if capabilities:
            self._capabilities = capabilities
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state["output_data"] = {"result": "processed"}
        return state


class TestProcessNode:
    """Test ProcessNode functionality."""
    
    def test_process_node_initialization(self):
        """Test process node initialization with processor."""
        mock_processor = MockProcessor({})
        node = ProcessNode(mock_processor)
        
        assert node.processor == mock_processor
    
    def test_process_node_execution_success(self):
        """Test successful process node execution."""
        mock_processor = MockProcessor({})
        node = ProcessNode(mock_processor)
        
        # Initial state
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['status'] == 'success'
        assert result['output_data'] == {"result": "processed"}
        assert result['error'] is None
    
    def test_process_node_execution_error(self):
        """Test process node execution with error."""
        mock_processor = MockProcessor({})
        
        # Override to raise exception
        def error_logic(state):
            raise ValueError("Processing failed")
        
        mock_processor._process_core_logic = error_logic
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        assert result['status'] == 'error'
        assert result['error'] is not None
        assert isinstance(result['error'], ValueError)
        assert "Processing failed" in str(result['error'])
    
    def test_process_node_with_observable_capability(self):
        """Test process node with observable processor."""
        mock_processor = MockProcessor({}, capabilities=['observable'])
        mock_processor._capabilities = ['observable']
        mock_processor._capability_configs = {
            'observable': {'observers': ['test_observer']}
        }
        
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should print monitoring events
            assert mock_print.called
            mock_print.assert_any_call("[MONITOR:test_observer] {'event_type': 'process_completed', 'node_name': 'process_core', 'processor_type': 'data_fetcher', 'state': {'status': 'success', 'retry_count': 0, 'has_output': True}}")
    
    def test_process_node_error_with_observable_capability(self):
        """Test process node error handling with observable processor."""
        mock_processor = MockProcessor({}, capabilities=['observable'])
        mock_processor._capabilities = ['observable']
        mock_processor._capability_configs = {
            'observable': {'observers': ['test_observer']}
        }
        
        # Override to raise exception
        def error_logic(state):
            raise ValueError("Processing failed")
        
        mock_processor._process_core_logic = error_logic
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should print error monitoring event
            assert mock_print.called
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("process_failed" in call for call in calls)
    
    def test_process_node_state_updates(self):
        """Test that process node properly updates state."""
        mock_processor = MockProcessor({})
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': ValueError("Previous error"),  # Should be cleared
            'status': 'error',  # Should be updated
            'retry_count': 2,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        result = node.execute(state)
        
        # Should clear error and update status
        assert result['error'] is None
        assert result['status'] == 'success'
        # Should preserve other fields
        assert result['retry_count'] == 2
        assert result['interrupt_requested'] is False
    
    def test_process_node_monitoring_event_structure(self):
        """Test monitoring event data structure."""
        mock_processor = MockProcessor({})
        mock_processor._capabilities = ['observable']
        mock_processor._capability_configs = {
            'observable': {'observers': ['test_observer']}
        }
        
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        # Test monitoring event structure
        with patch('builtins.print') as mock_print:
            node.execute(state)
            
            # Check that print was called with monitoring event
            assert mock_print.called
            call_args = str(mock_print.call_args_list[0])
            
            # Should contain required fields
            assert "event_type" in call_args
            assert "node_name" in call_args
            assert "processor_type" in call_args
            assert "state" in call_args
    
    def test_process_node_without_observable_capability(self):
        """Test process node without observable capability."""
        mock_processor = MockProcessor({})  # No observable capability
        node = ProcessNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'initialized',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.execute(state)
            
            # Should not print monitoring events
            assert not mock_print.called
            assert result['status'] == 'success' 