"""
Tests for src/core/decorators/evaluable.py

Tests the @evaluable decorator functionality:
- Capability registration
- Retry configuration
- Method requirements
"""

import pytest
from unittest.mock import Mock, patch

from src.core.decorators.evaluable import evaluable
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState


class TestEvaluableDecorator:
    """Test the @evaluable decorator functionality."""
    
    def test_evaluable_basic(self):
        """Test basic evaluable decorator application."""
        @evaluable(max_retries=3)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Check capability registration
        assert processor.has_capability('evaluable')
        assert 'evaluable' in processor.get_capabilities()
        
        # Check configuration
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 3
    
    def test_evaluable_with_timeout(self):
        """Test evaluable with timeout configuration."""
        @evaluable(max_retries=2, timeout_seconds=30)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 2
        assert config['timeout_seconds'] == 30
    
    def test_evaluable_default_values(self):
        """Test evaluable decorator with default values."""
        @evaluable(max_retries=1)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 1
        assert 'timeout_seconds' in config
        assert config['timeout_seconds'] == 60  # Default value
    
    def test_evaluable_requires_evaluate_method(self):
        """Test that evaluable requires _evaluate_result method."""
        with pytest.raises(NotImplementedError):
            @evaluable(max_retries=1)
            class BadProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                # Missing _evaluate_result method
            
            BadProcessor({})
    
    def test_evaluable_zero_retries(self):
        """Test evaluable with zero retries."""
        @evaluable(max_retries=0)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 0
    
    def test_evaluable_negative_retries_invalid(self):
        """Test that negative retries are invalid."""
        with pytest.raises(ValueError):
            @evaluable(max_retries=-1)
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
    
    def test_evaluable_with_retry_strategy(self):
        """Test evaluable with retry strategy configuration."""
        @evaluable(max_retries=3, retry_strategy="exponential_backoff")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['retry_strategy'] == "exponential_backoff"
    
    def test_evaluable_inheritance(self):
        """Test evaluable decorator with inheritance."""
        @evaluable(max_retries=2)
        class BaseProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        # Child class should inherit the capability
        class ChildProcessor(BaseProcessor):
            pass
        
        child = ChildProcessor({})
        
        assert child.has_capability('evaluable')
        config = child.get_capability_config('evaluable')
        assert config['max_retries'] == 2
    
    def test_evaluable_with_other_decorators(self):
        """Test evaluable combined with other decorators."""
        from src.core.decorators.observable import observable
        
        @evaluable(max_retries=2)
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Should have both capabilities
        assert processor.has_capability('evaluable')
        assert processor.has_capability('observable')
        
        capabilities = processor.get_capabilities()
        assert 'evaluable' in capabilities
        assert 'observable' in capabilities
    
    def test_evaluable_high_retry_count(self):
        """Test evaluable with high retry count."""
        @evaluable(max_retries=100)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 100
    
    def test_evaluable_custom_evaluation_method(self):
        """Test evaluable with custom evaluation method name."""
        @evaluable(max_retries=1, evaluation_method="custom_evaluate")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('evaluable')
        assert config['evaluation_method'] == "custom_evaluate"
    
    def test_evaluable_config_immutability(self):
        """Test that evaluable configuration is immutable after creation."""
        @evaluable(max_retries=3)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Get configuration
        config = processor.get_capability_config('evaluable')
        original_retries = config['max_retries']
        
        # Try to modify (should not affect original)
        config['max_retries'] = 999
        
        # Get fresh configuration
        fresh_config = processor.get_capability_config('evaluable')
        assert fresh_config['max_retries'] == original_retries
    
    def test_evaluable_multiple_instances(self):
        """Test multiple instances of evaluable processors."""
        @evaluable(max_retries=5)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor1 = TestProcessor({})
        processor2 = TestProcessor({})
        
        # Both should have the same capabilities but be independent
        assert processor1.has_capability('evaluable')
        assert processor2.has_capability('evaluable')
        assert processor1 is not processor2
        
        config1 = processor1.get_capability_config('evaluable')
        config2 = processor2.get_capability_config('evaluable')
        
        assert config1 == config2
        assert config1 is not config2  # Should be separate instances 