"""
Tests for src/core/workflow_nodes/interrupt_node.py

Tests the interrupt control node functionality:
- Node initialization with processor
- Interrupt detection and handling
- Checkpoint creation and restoration
- Monitoring integration
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.workflow_nodes.interrupt_node import InterruptNode
from src.core.base_processor import BaseProcessor, ProcessorState, ProcessorType


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def __init__(self, config: Dict[str, Any], capabilities: list = None):
        super().__init__(config)
        if capabilities:
            self._capabilities = capabilities
            # Set capability configs
            if 'interruptible' in capabilities:
                self._capability_configs['interruptible'] = config.get('interrupt_config', {'save_point_id': 'default_save_point'})
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state["output_data"] = {"result": "processed"}
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """Default interrupt handling implementation."""
        state["status"] = "interrupted"
        return state


class TestInterruptNode:
    """Test InterruptNode functionality."""
    
    def test_interrupt_node_initialization(self):
        """Test interrupt node initialization with processor."""
        mock_processor = MockProcessor({})
        node = InterruptNode(mock_processor)
        
        assert node.processor == mock_processor
    
    def test_check_interrupt_no_interrupt(self):
        """Test check_interrupt when no interrupt is requested."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': False,  # No interrupt
            'save_point_id': None
        }
        
        result = node.check_interrupt(state)
        
        assert result['status'] == 'continuing'
        assert result['interrupt_requested'] is False
    
    def test_check_interrupt_with_interrupt(self):
        """Test check_interrupt when interrupt is requested."""
        mock_processor = MockProcessor(
            {'interrupt_config': {'save_point_id': 'test_save_point'}}, 
            capabilities=['interruptible']
        )
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': True,  # Interrupt requested
            'save_point_id': None
        }
        
        result = node.check_interrupt(state)
        
        assert result['status'] == 'interrupt_requested'
        assert result['save_point_id'] == 'test_save_point'
    
    def test_handle_interrupt_success(self):
        """Test successful interrupt handling."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'interrupt_requested',
            'retry_count': 0,
            'interrupt_requested': True,
            'save_point_id': None
        }
        
        result = node.handle_interrupt(state)
        
        assert result['status'] == 'interrupted'
    
    def test_handle_interrupt_error(self):
        """Test interrupt handling with error."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])
        
        # Override to raise exception in interrupt handling
        def error_interrupt(state):
            raise RuntimeError("Interrupt handling failed")
        
        mock_processor._handle_interrupt = error_interrupt
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'interrupt_requested',
            'retry_count': 0,
            'interrupt_requested': True,
            'save_point_id': None
        }
        
        result = node.handle_interrupt(state)
        
        assert result['interrupt_error'] is not None
        assert isinstance(result['interrupt_error'], RuntimeError)
        assert result['status'] == 'interrupt_handling_failed'
    
    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'partial'},
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': 'test_save_point'
        }
        
        result = node.create_checkpoint(state)
        
        assert result['checkpoint_created'] is True
        assert 'checkpoint_timestamp' in result
    
    def test_resume_from_checkpoint(self):
        """Test resuming from checkpoint."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])
        node = InterruptNode(mock_processor)
        
        restored_state = node.resume_from_checkpoint('test_save_point')
        
        assert restored_state['status'] == 'resumed_from_checkpoint'
        assert restored_state['save_point_id'] == 'test_save_point'
    
    def test_interrupt_with_observable_capability(self):
        """Test interrupt operations with observable processor."""
        mock_processor = MockProcessor({}, capabilities=['interruptible', 'observable'])
        mock_processor._capability_configs = {
            'interruptible': {'save_point_id': 'test_save_point'},
            'observable': {'observers': ['test_observer']}
        }
        
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': True,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            node.check_interrupt(state)
            
            # Should print monitoring events
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("interrupt_detected" in call for call in calls)
    
    def test_handle_interrupt_with_monitoring(self):
        """Test interrupt handling with monitoring events."""
        mock_processor = MockProcessor({}, capabilities=['interruptible', 'observable'])
        mock_processor._capability_configs = {
            'interruptible': {'save_point_id': 'test_save_point'},
            'observable': {'observers': ['test_observer']}
        }
        
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'interrupt_requested',
            'retry_count': 0,
            'interrupt_requested': True,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            node.handle_interrupt(state)
            
            # Should print monitoring events
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("interrupt_handling_started" in call for call in calls)
            assert any("interrupt_handling_completed" in call for call in calls)
    
    def test_checkpoint_with_monitoring(self):
        """Test checkpoint operations with monitoring events."""
        mock_processor = MockProcessor({}, capabilities=['interruptible', 'observable'])
        mock_processor._capability_configs = {
            'interruptible': {'save_point_id': 'test_save_point'},
            'observable': {'observers': ['test_observer']}
        }
        
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': {'result': 'partial'},
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': False,
            'save_point_id': 'test_save_point'
        }
        
        with patch('builtins.print') as mock_print:
            node.create_checkpoint(state)
            
            # Should print checkpoint creation event
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("checkpoint_created" in call for call in calls)
    
    def test_resume_with_monitoring(self):
        """Test resume operations with monitoring events."""
        mock_processor = MockProcessor({}, capabilities=['interruptible', 'observable'])
        mock_processor._capability_configs = {
            'interruptible': {'save_point_id': 'test_save_point'},
            'observable': {'observers': ['test_observer']}
        }
        
        node = InterruptNode(mock_processor)
        
        with patch('builtins.print') as mock_print:
            node.resume_from_checkpoint('test_save_point')
            
            # Should print resume event
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("resumed_from_checkpoint" in call for call in calls)
    
    def test_interrupt_without_observable_capability(self):
        """Test interrupt operations without observable capability."""
        mock_processor = MockProcessor({}, capabilities=['interruptible'])  # No observable
        node = InterruptNode(mock_processor)
        
        state: ProcessorState = {
            'input_data': {'test': 'data'},
            'output_data': None,
            'error': None,
            'status': 'processing',
            'retry_count': 0,
            'interrupt_requested': True,
            'save_point_id': None
        }
        
        with patch('builtins.print') as mock_print:
            result = node.check_interrupt(state)
            
            # Should not print monitoring events
            assert not mock_print.called
            assert result['status'] == 'interrupt_requested' 