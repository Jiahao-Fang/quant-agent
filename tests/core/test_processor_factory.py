"""
Tests for src/core/processor_factory.py

Tests ProcessorFactory with new simplified architecture:
- Processor creation and validation
- Capability detection
- Subgraph generation
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.core.processor_factory import ProcessorFactory, ProcessorConfig
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState, ProcessorResult
from src.core.decorators import observable, evaluable, debuggable, interruptible


# Test processor classes
class BasicTestProcessor(BaseProcessor):
    """Basic processor for testing factory."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"basic": "result"}
        state['status'] = 'success'
        return state


@observable(observers=["test_observer"])
@evaluable(max_retries=2)
class AdvancedTestProcessor(BaseProcessor):
    """Advanced processor with capabilities for testing factory."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.FEATURE_BUILDER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"advanced": "result"}
        state['status'] = 'success'
        return state
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        state['eval_passed'] = True
        return state


@interruptible(save_point_id="test_save_point")
class InterruptibleTestProcessor(BaseProcessor):
    """Interruptible processor for testing factory."""
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.BACKTEST_RUNNER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        state['output_data'] = {"interruptible": "result"}
        state['status'] = 'success'
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        state['status'] = 'paused'
        return state


class TestProcessorFactory:
    """Test ProcessorFactory functionality."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = ProcessorFactory()
        
        assert hasattr(factory, '_processor_classes')
        assert factory._processor_classes == {}
        assert hasattr(factory, '_subgraph_builder')
    
    def test_register_processor_class(self):
        """Test registering processor classes."""
        factory = ProcessorFactory()
        
        # Register basic processor
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        assert ProcessorType.DATA_FETCHER in factory._processor_classes
        assert factory._processor_classes[ProcessorType.DATA_FETCHER] == BasicTestProcessor
    
    def test_register_multiple_processors(self):
        """Test registering multiple processor classes."""
        factory = ProcessorFactory()
        
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        assert len(factory._processor_classes) == 2
        assert factory._processor_classes[ProcessorType.DATA_FETCHER] == BasicTestProcessor
        assert factory._processor_classes[ProcessorType.FEATURE_BUILDER] == AdvancedTestProcessor
    
    def test_register_invalid_class(self):
        """Test registering invalid processor class."""
        factory = ProcessorFactory()
        
        class NotProcessor:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseProcessor"):
            factory.register_processor_class(ProcessorType.DATA_FETCHER, NotProcessor)
    
    def test_create_processor_basic(self):
        """Test creating a basic processor."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        config = {"param": "value"}
        processor = factory.create_processor(ProcessorType.DATA_FETCHER, config)
        
        assert isinstance(processor, BasicTestProcessor)
        assert processor.config == config
        assert processor.get_processor_type() == ProcessorType.DATA_FETCHER
    
    def test_create_processor_with_capabilities(self):
        """Test creating a processor with capabilities."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        processor = factory.create_processor(ProcessorType.FEATURE_BUILDER, {})
        
        assert isinstance(processor, AdvancedTestProcessor)
        assert processor.has_capability('observable')
        assert processor.has_capability('evaluable')
        assert not processor.has_capability('debuggable')
    
    def test_create_processor_unregistered_type(self):
        """Test creating processor with unregistered type."""
        factory = ProcessorFactory()
        
        with pytest.raises(ValueError, match="not registered"):
            factory.create_processor(ProcessorType.DATA_FETCHER, {})
    
    def test_get_available_processors(self):
        """Test getting available processor types."""
        factory = ProcessorFactory()
        
        # Initially empty
        assert factory.get_available_processors() == []
        
        # After registration
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        available_types = factory.get_available_processors()
        assert ProcessorType.DATA_FETCHER in available_types
        assert ProcessorType.FEATURE_BUILDER in available_types
        assert len(available_types) == 2
    
    def test_is_processor_registered(self):
        """Test checking if processor type is registered."""
        factory = ProcessorFactory()
        
        assert not factory.is_processor_registered(ProcessorType.DATA_FETCHER)
        
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        assert factory.is_processor_registered(ProcessorType.DATA_FETCHER)
        assert not factory.is_processor_registered(ProcessorType.FEATURE_BUILDER)
    
    def test_get_processor_capabilities(self):
        """Test getting processor capabilities."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        capabilities = factory.get_processor_capabilities(ProcessorType.FEATURE_BUILDER)
        
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
    
    def test_get_processor_capabilities_unregistered(self):
        """Test getting capabilities for unregistered processor."""
        factory = ProcessorFactory()
        
        with pytest.raises(ValueError, match="not registered"):
            factory.get_processor_capabilities(ProcessorType.DATA_FETCHER)
    
    def test_unregister_processor(self):
        """Test unregistering processor type."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        assert factory.is_processor_registered(ProcessorType.DATA_FETCHER)
        
        factory.unregister_processor(ProcessorType.DATA_FETCHER)
        
        assert not factory.is_processor_registered(ProcessorType.DATA_FETCHER)
    
    def test_clear_all_registrations(self):
        """Test clearing all processor registrations."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        assert len(factory.get_available_processors()) == 2
        
        factory.clear_all_registrations()
        
        assert len(factory.get_available_processors()) == 0


class TestProcessorConfig:
    """Test ProcessorConfig dataclass."""
    
    def test_processor_config_creation(self):
        """Test creating ProcessorConfig."""
        config = ProcessorConfig(
            processor_type=ProcessorType.DATA_FETCHER,
            config_params={"param": "value"}
        )
        
        assert config.processor_type == ProcessorType.DATA_FETCHER
        assert config.config_params == {"param": "value"}
    
    def test_processor_config_none_params(self):
        """Test ProcessorConfig with None config_params."""
        config = ProcessorConfig(
            processor_type=ProcessorType.DATA_FETCHER,
            config_params=None
        )
        
        assert config.config_params == {}
    
    def test_create_processor_with_config(self):
        """Test creating processor with ProcessorConfig."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        config = ProcessorConfig(
            processor_type=ProcessorType.DATA_FETCHER,
            config_params={"param": "value"}
        )
        
        processor = factory.create_processor_with_config(config)
        
        assert isinstance(processor, BasicTestProcessor)
        assert processor.config == {"param": "value"}


class TestSubgraphGeneration:
    """Test subgraph generation functionality."""
    
    @patch('src.core.processor_factory.SubgraphBuilder')
    def test_create_processor_subgraph(self, mock_subgraph_builder):
        """Test creating processor subgraph."""
        factory = ProcessorFactory()
        processor = BasicTestProcessor({})
        
        # Mock the subgraph builder
        mock_builder_instance = Mock()
        mock_subgraph = Mock()
        mock_builder_instance.create_subgraph.return_value = mock_subgraph
        factory._subgraph_builder = mock_builder_instance
        
        subgraph = factory.create_processor_subgraph(processor)
        
        mock_builder_instance.create_subgraph.assert_called_once_with(processor)
        assert subgraph == mock_subgraph
    
    @patch('src.core.processor_factory.SubgraphBuilder')
    def test_create_subgraph_with_capabilities(self, mock_subgraph_builder):
        """Test creating subgraph for processor with capabilities."""
        factory = ProcessorFactory()
        processor = AdvancedTestProcessor({})
        
        mock_builder_instance = Mock()
        mock_subgraph = Mock()
        mock_builder_instance.create_subgraph.return_value = mock_subgraph
        factory._subgraph_builder = mock_builder_instance
        
        subgraph = factory.create_processor_subgraph(processor)
        
        mock_builder_instance.create_subgraph.assert_called_once_with(processor)
        assert subgraph == mock_subgraph
    
    def test_subgraph_creation_error(self):
        """Test subgraph creation error handling."""
        factory = ProcessorFactory()
        processor = BasicTestProcessor({})
        
        # Mock builder to raise exception
        mock_builder = Mock()
        mock_builder.create_subgraph.side_effect = RuntimeError("Subgraph creation failed")
        factory._subgraph_builder = mock_builder
        
        with pytest.raises(RuntimeError, match="Subgraph creation failed"):
            factory.create_processor_subgraph(processor)


class TestCapabilityValidation:
    """Test capability validation functionality."""
    
    def test_validate_processor_capabilities_basic(self):
        """Test validating basic processor capabilities."""
        factory = ProcessorFactory()
        processor = BasicTestProcessor({})
        
        # Should not raise any exceptions
        assert factory.validate_processor_capabilities(processor) is True
    
    def test_validate_processor_capabilities_with_decorators(self):
        """Test validating processor with capability decorators."""
        factory = ProcessorFactory()
        processor = AdvancedTestProcessor({})
        
        # Should validate that required methods are implemented
        assert factory.validate_processor_capabilities(processor) is True
    
    def test_validate_processor_missing_methods(self):
        """Test validation fails for missing required methods."""
        
        @evaluable(max_retries=1)
        class BadProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            # Missing _evaluate_result method
        
        factory = ProcessorFactory()
        
        with pytest.raises(NotImplementedError):
            BadProcessor({})


class TestFactoryStatus:
    """Test factory status and management functionality."""
    
    def test_get_factory_status(self):
        """Test getting factory status."""
        factory = ProcessorFactory()
        
        status = factory.get_factory_status()
        
        assert 'registered_processors' in status
        assert 'available_types' in status
        assert 'processor_details' in status
        assert status['registered_processors'] == 0
    
    def test_get_factory_status_with_processors(self):
        """Test getting factory status with registered processors."""
        factory = ProcessorFactory()
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        
        status = factory.get_factory_status()
        
        assert status['registered_processors'] == 2
        assert 'data_fetcher' in status['available_types']
        assert 'feature_builder' in status['available_types']
        
        # Check processor details
        details = status['processor_details']
        assert 'data_fetcher' in details
        assert 'feature_builder' in details
        
        basic_details = details['data_fetcher']
        assert basic_details['class_name'] == 'BasicTestProcessor'
        assert basic_details['capabilities'] == []
        assert basic_details['has_capabilities'] is False
        
        advanced_details = details['feature_builder']
        assert advanced_details['class_name'] == 'AdvancedTestProcessor'
        assert 'observable' in advanced_details['capabilities']
        assert 'evaluable' in advanced_details['capabilities']
        assert advanced_details['has_capabilities'] is True


class TestFactoryIntegration:
    """Test factory integration scenarios."""
    
    def test_end_to_end_processor_creation(self):
        """Test complete processor creation workflow."""
        factory = ProcessorFactory()
        
        # Register processor
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        # Create processor
        config = {"param": "value"}
        processor = factory.create_processor(ProcessorType.DATA_FETCHER, config)
        
        # Validate processor
        assert factory.validate_processor_capabilities(processor)
        
        # Create subgraph
        mock_builder = Mock()
        mock_subgraph = Mock()
        mock_builder.create_subgraph.return_value = mock_subgraph
        factory._subgraph_builder = mock_builder
        
        subgraph = factory.create_processor_subgraph(processor)
        
        assert subgraph == mock_subgraph
    
    def test_factory_with_multiple_processor_types(self):
        """Test factory with multiple processor types."""
        factory = ProcessorFactory()
        
        # Register multiple processors
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        factory.register_processor_class(ProcessorType.FEATURE_BUILDER, AdvancedTestProcessor)
        factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, InterruptibleTestProcessor)
        
        # Create processors of different types
        basic = factory.create_processor(ProcessorType.DATA_FETCHER, {})
        advanced = factory.create_processor(ProcessorType.FEATURE_BUILDER, {})
        interruptible = factory.create_processor(ProcessorType.BACKTEST_RUNNER, {})
        
        # Verify different capabilities
        assert basic.get_capabilities() == []
        assert 'observable' in advanced.get_capabilities()
        assert 'evaluable' in advanced.get_capabilities()
        assert 'interruptible' in interruptible.get_capabilities()
    
    def test_factory_error_handling(self):
        """Test factory error handling."""
        factory = ProcessorFactory()
        
        # Test various error conditions
        with pytest.raises(ValueError):
            factory.create_processor(ProcessorType.DATA_FETCHER, {})
        
        # Register and test again
        factory.register_processor_class(ProcessorType.DATA_FETCHER, BasicTestProcessor)
        
        # This should work
        processor = factory.create_processor(ProcessorType.DATA_FETCHER, {})
        assert processor is not None 