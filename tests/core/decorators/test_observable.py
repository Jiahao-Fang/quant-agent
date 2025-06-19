"""
Tests for src/core/decorators/observable.py

Tests the @observable decorator functionality:
- Capability registration
- Observer configuration
- Method requirements
"""

import pytest
from unittest.mock import Mock, patch

from src.core.decorators.observable import observable
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState


class TestObservableDecorator:
    """Test the @observable decorator functionality."""
    
    def test_observable_basic(self):
        """Test basic observable decorator application."""
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Check capability registration
        assert processor.has_capability('observable')
        assert 'observable' in processor.get_capabilities()
        
        # Check configuration
        config = processor.get_capability_config('observable')
        assert config['observers'] == ["test_observer"]
    
    def test_observable_multiple_observers(self):
        """Test observable with multiple observers."""
        @observable(observers=["observer1", "observer2", "observer3"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('observable')
        assert config['observers'] == ["observer1", "observer2", "observer3"]
        assert len(config['observers']) == 3
    
    def test_observable_with_callback_method(self):
        """Test observable with callback method parameter."""
        @observable(observers=["test_observer"], callback_method="custom_callback")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('observable')
        assert config['callback_method'] == "custom_callback"
    
    def test_observable_default_values(self):
        """Test observable decorator with default values."""
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('observable')
        assert 'callback_method' in config
        assert config['callback_method'] == "on_state_change"  # Default value
    
    def test_observable_empty_observers(self):
        """Test observable with empty observers list."""
        @observable(observers=[])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('observable')
        assert config['observers'] == []
    
    def test_observable_inheritance(self):
        """Test observable decorator with inheritance."""
        @observable(observers=["base_observer"])
        class BaseProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        # Child class should inherit the capability
        class ChildProcessor(BaseProcessor):
            pass
        
        child = ChildProcessor({})
        
        assert child.has_capability('observable')
        config = child.get_capability_config('observable')
        assert config['observers'] == ["base_observer"]
    
    def test_observable_with_other_decorators(self):
        """Test observable combined with other decorators."""
        from src.core.decorators.evaluable import evaluable
        
        @observable(observers=["test_observer"])
        @evaluable(max_retries=2)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Should have both capabilities
        assert processor.has_capability('observable')
        assert processor.has_capability('evaluable')
        
        capabilities = processor.get_capabilities()
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
    
    def test_observable_invalid_observers_type(self):
        """Test observable with invalid observers type."""
        with pytest.raises(TypeError):
            @observable(observers="invalid_type")  # Should be list
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
    
    def test_observable_config_immutability(self):
        """Test that observable configuration is immutable after creation."""
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Get configuration
        config = processor.get_capability_config('observable')
        original_observers = config['observers'].copy()
        
        # Try to modify (should not affect original)
        config['observers'].append("new_observer")
        
        # Get fresh configuration
        fresh_config = processor.get_capability_config('observable')
        assert fresh_config['observers'] == original_observers
    
    def test_observable_multiple_instances(self):
        """Test multiple instances of observable processors."""
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor1 = TestProcessor({})
        processor2 = TestProcessor({})
        
        # Both should have the same capabilities but be independent
        assert processor1.has_capability('observable')
        assert processor2.has_capability('observable')
        assert processor1 is not processor2
        
        config1 = processor1.get_capability_config('observable')
        config2 = processor2.get_capability_config('observable')
        
        assert config1 == config2
        assert config1 is not config2  # Should be separate instances 