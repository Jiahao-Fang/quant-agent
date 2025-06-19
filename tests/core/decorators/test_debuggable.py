"""
Tests for src/core/decorators/debuggable.py

Tests the @debuggable decorator functionality:
- Capability registration
- Debug configuration
- Method requirements
"""

import pytest
from unittest.mock import Mock, patch

from src.core.decorators.debuggable import debuggable
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState


class TestDebuggableDecorator:
    """Test the @debuggable decorator functionality."""
    
    def test_debuggable_basic(self):
        """Test basic debuggable decorator application."""
        @debuggable(debug_level="INFO")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Check capability registration
        assert processor.has_capability('debuggable')
        assert 'debuggable' in processor.get_capabilities()
        
        # Check configuration
        config = processor.get_capability_config('debuggable')
        assert config['debug_level'] == "INFO"
    
    def test_debuggable_with_log_file(self):
        """Test debuggable with log file configuration."""
        @debuggable(debug_level="DEBUG", log_file="debug.log")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('debuggable')
        assert config['debug_level'] == "DEBUG"
        assert config['log_file'] == "debug.log"
    
    def test_debuggable_default_values(self):
        """Test debuggable decorator with default values."""
        @debuggable(debug_level="WARNING")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('debuggable')
        assert config['debug_level'] == "WARNING"
        assert 'log_file' in config
        assert config['log_file'] is None  # Default value
    
    def test_debuggable_requires_debug_method(self):
        """Test that debuggable requires _collect_debug_info method."""
        with pytest.raises(NotImplementedError):
            @debuggable(debug_level="INFO")
            class BadProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                # Missing _collect_debug_info method
            
            BadProcessor({})
    
    def test_debuggable_all_debug_levels(self):
        """Test debuggable with all debug levels."""
        debug_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in debug_levels:
            @debuggable(debug_level=level)
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                
                def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                    return state
            
            processor = TestProcessor({})
            config = processor.get_capability_config('debuggable')
            assert config['debug_level'] == level
    
    def test_debuggable_invalid_debug_level(self):
        """Test debuggable with invalid debug level."""
        with pytest.raises(ValueError):
            @debuggable(debug_level="INVALID_LEVEL")
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
    
    def test_debuggable_with_enable_profiling(self):
        """Test debuggable with profiling enabled."""
        @debuggable(debug_level="DEBUG", enable_profiling=True)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('debuggable')
        assert config['enable_profiling'] is True
    
    def test_debuggable_inheritance(self):
        """Test debuggable decorator with inheritance."""
        @debuggable(debug_level="INFO")
        class BaseProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        # Child class should inherit the capability
        class ChildProcessor(BaseProcessor):
            pass
        
        child = ChildProcessor({})
        
        assert child.has_capability('debuggable')
        config = child.get_capability_config('debuggable')
        assert config['debug_level'] == "INFO"
    
    def test_debuggable_with_other_decorators(self):
        """Test debuggable combined with other decorators."""
        from src.core.decorators.observable import observable
        
        @debuggable(debug_level="DEBUG")
        @observable(observers=["test_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Should have both capabilities
        assert processor.has_capability('debuggable')
        assert processor.has_capability('observable')
        
        capabilities = processor.get_capabilities()
        assert 'debuggable' in capabilities
        assert 'observable' in capabilities
    
    def test_debuggable_with_custom_debug_method(self):
        """Test debuggable with custom debug method name."""
        @debuggable(debug_level="INFO", debug_method="custom_debug")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('debuggable')
        assert config['debug_method'] == "custom_debug"
    
    def test_debuggable_with_memory_tracking(self):
        """Test debuggable with memory tracking enabled."""
        @debuggable(debug_level="DEBUG", track_memory=True)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        config = processor.get_capability_config('debuggable')
        assert config['track_memory'] is True
    
    def test_debuggable_config_immutability(self):
        """Test that debuggable configuration is immutable after creation."""
        @debuggable(debug_level="ERROR")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        # Get configuration
        config = processor.get_capability_config('debuggable')
        original_level = config['debug_level']
        
        # Try to modify (should not affect original)
        config['debug_level'] = "CRITICAL"
        
        # Get fresh configuration
        fresh_config = processor.get_capability_config('debuggable')
        assert fresh_config['debug_level'] == original_level
    
    def test_debuggable_multiple_instances(self):
        """Test multiple instances of debuggable processors."""
        @debuggable(debug_level="WARNING")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _collect_debug_info(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor1 = TestProcessor({})
        processor2 = TestProcessor({})
        
        # Both should have the same capabilities but be independent
        assert processor1.has_capability('debuggable')
        assert processor2.has_capability('debuggable')
        assert processor1 is not processor2
        
        config1 = processor1.get_capability_config('debuggable')
        config2 = processor2.get_capability_config('debuggable')
        
        assert config1 == config2
        assert config1 is not config2  # Should be separate instances 