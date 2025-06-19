"""
Tests for src/core/decorators/interruptible.py

Tests the @interruptible decorator functionality:
- Capability registration
- Checkpoint configuration
- Method requirements
"""

import pytest
from unittest.mock import Mock, patch

from src.core.decorators.interruptible import interruptible
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState


class TestInterruptibleDecorator:
    """Test the @interruptible decorator functionality."""
    
    def test_interruptible_basic(self):
        """Test basic interruptible decorator application."""
        @interruptible(save_point_id="test_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Check capability registration
        assert processor.has_capability('interruptible')
        assert 'interruptible' in processor.get_capabilities()
        
        # Check configuration
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "test_save_point"
    
    def test_interruptible_with_save_frequency(self):
        """Test interruptible with save frequency configuration."""
        @interruptible(save_point_id="test_save_point", save_frequency=100)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "test_save_point"
        assert config['save_frequency'] == 100
    
    def test_interruptible_default_values(self):
        """Test interruptible decorator with default values."""
        @interruptible(save_point_id="default_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "default_save_point"
        assert 'save_frequency' in config
        assert config['save_frequency'] == 10  # Default value
    
    def test_interruptible_requires_interrupt_method(self):
        """Test that interruptible requires _handle_interrupt method."""
        with pytest.raises(NotImplementedError):
            @interruptible(save_point_id="test_save_point")
            class BadProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                # Missing _handle_interrupt method
            
            BadProcessor({})
    
    def test_interruptible_with_auto_resume(self):
        """Test interruptible with auto resume enabled."""
        @interruptible(save_point_id="auto_save_point", auto_resume=True)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['auto_resume'] is True
    
    def test_interruptible_with_timeout(self):
        """Test interruptible with interrupt timeout."""
        @interruptible(save_point_id="timeout_save_point", interrupt_timeout=60)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['interrupt_timeout'] == 60
    
    def test_interruptible_empty_save_point_id(self):
        """Test interruptible with empty save_point ID."""
        with pytest.raises(ValueError):
            @interruptible(save_point_id="")
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
    
    def test_interruptible_none_save_point_id(self):
        """Test interruptible with None save_point ID."""
        with pytest.raises(TypeError):
            @interruptible(save_point_id=None)
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
    
    def test_interruptible_inheritance(self):
        """Test interruptible decorator with inheritance."""
        @interruptible(save_point_id="inherited_save_point")
        class BaseProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        # Child class should inherit the capability
        class ChildProcessor(BaseProcessor):
            pass
        
        child = ChildProcessor({})
        
        assert child.has_capability('interruptible')
        config = child.get_capability_config('interruptible')
        assert config['save_point_id'] == "inherited_save_point"
    
    def test_interruptible_with_other_decorators(self):
        """Test interruptible combined with other decorators."""
        from src.core.decorators.observable import observable
        
        @interruptible(save_point_id="multi_save_point")
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Should have both capabilities
        assert processor.has_capability('interruptible')
        assert processor.has_capability('observable')
        
        capabilities = processor.get_capabilities()
        assert 'interruptible' in capabilities
        assert 'observable' in capabilities
    
    def test_interruptible_with_custom_interrupt_method(self):
        """Test interruptible with custom interrupt method name."""
        @interruptible(save_point_id="custom_save_point", interrupt_method="custom_interrupt")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['interrupt_method'] == "custom_interrupt"
    
    def test_interruptible_high_save_frequency(self):
        """Test interruptible with high save frequency."""
        @interruptible(save_point_id="high_freq_save_point", save_frequency=1000)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['save_frequency'] == 1000
    
    def test_interruptible_zero_save_frequency(self):
        """Test interruptible with zero save frequency (save only on interrupt)."""
        @interruptible(save_point_id="zero_freq_save_point", save_frequency=0)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['save_frequency'] == 0
    
    def test_interruptible_config_immutability(self):
        """Test that interruptible configuration is immutable after creation."""
        @interruptible(save_point_id="immutable_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Get configuration
        config = processor.get_capability_config('interruptible')
        original_id = config['save_point_id']
        
        # Try to modify (should not affect original)
        config['save_point_id'] = "modified_save_point"
        
        # Get fresh configuration
        fresh_config = processor.get_capability_config('interruptible')
        assert fresh_config['save_point_id'] == original_id
    
    def test_interruptible_multiple_instances(self):
        """Test multiple instances of interruptible processors."""
        @interruptible(save_point_id="multi_instance_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor1 = TestProcessor({})
        processor2 = TestProcessor({})
        
        # Both should have the same capabilities but be independent
        assert processor1.has_capability('interruptible')
        assert processor2.has_capability('interruptible')
        assert processor1 is not processor2
        
        config1 = processor1.get_capability_config('interruptible')
        config2 = processor2.get_capability_config('interruptible')
        
        assert config1 == config2
        assert config1 is not config2  # Should be separate instances
    
    def test_interruptible_with_all_options(self):
        """Test interruptible with all configuration options."""
        @interruptible(
            save_point_id="full_config_save_point",
            save_frequency=50,
            auto_resume=True,
            interrupt_timeout=120,
            interrupt_method="full_interrupt"
        )
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "full_config_save_point"
        assert config['save_frequency'] == 50
        assert config['auto_resume'] is True
        assert config['interrupt_timeout'] == 120
        assert config['interrupt_method'] == "full_interrupt" 