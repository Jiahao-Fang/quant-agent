"""
Tests for src/core/decorators/ modules

Tests decorator functionality:
- Observable decorator
- Evaluable decorator  
- Debuggable decorator
- Interruptible decorator
- Multiple decorator combinations
"""

import pytest
from typing import Dict, Any

from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from src.core.decorators import observable, evaluable, debuggable, interruptible


class TestDecoratorBasics:
    """Test basic decorator functionality."""
    
    def test_observable_decorator(self):
        """Test @observable decorator."""
        @observable(observers=["ui", "logger"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        # Check capability is recorded
        assert 'observable' in processor.get_capabilities()
        # Check configuration
        config = processor.get_capability_config('observable')
        assert config['observers'] == ["ui", "logger"]
    
    def test_evaluable_decorator(self):
        """Test @evaluable decorator."""
        @evaluable(max_retries=5)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        assert 'evaluable' in processor.get_capabilities()
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 5
    
    def test_debuggable_decorator(self):
        """Test @debuggable decorator."""
        @debuggable(max_retries=3)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        assert 'debuggable' in processor.get_capabilities()
        config = processor.get_capability_config('debuggable')
        assert config['max_retries'] == 3
    
    def test_interruptible_decorator(self):
        """Test @interruptible decorator."""
        @interruptible(save_point_id="test_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        assert 'interruptible' in processor.get_capabilities()
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "test_save_point"


class TestMultipleDecorators:
    """Test processors with multiple decorators."""
    
    def test_all_decorators_combined(self):
        """Test processor with all decorators."""
        @observable(observers=["monitor"])
        @evaluable(max_retries=2)
        @debuggable(max_retries=1)
        @interruptible(save_point_id="combined")
        class FullProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = FullProcessor({})
        capabilities = processor.get_capabilities()
        
        assert len(capabilities) == 4
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
        
        # Check individual configurations
        assert processor.get_capability_config('observable')['observers'] == ["monitor"]
        assert processor.get_capability_config('evaluable')['max_retries'] == 2
        assert processor.get_capability_config('debuggable')['max_retries'] == 1
        assert processor.get_capability_config('interruptible')['save_point_id'] == "combined"
    
    def test_partial_decorators(self):
        """Test processor with only some decorators."""
        @observable(observers=["partial"])
        @evaluable(max_retries=4)
        class PartialProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = PartialProcessor({})
        capabilities = processor.get_capabilities()
        
        assert len(capabilities) == 2
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' not in capabilities
        assert 'interruptible' not in capabilities


class TestDecoratorValidation:
    """Test decorator validation logic."""
    
    def test_evaluable_without_method_raises_error(self):
        """Test that @evaluable without _evaluate_result raises error."""
        @evaluable(max_retries=1)
        class BadProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            # Missing _evaluate_result method
        
        with pytest.raises(NotImplementedError):
            BadProcessor({})
    
    def test_debuggable_without_method_raises_error(self):
        """Test that @debuggable without _debug_error raises error."""
        @debuggable(max_retries=1)
        class BadProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            # Missing _debug_error method
        
        with pytest.raises(NotImplementedError):
            BadProcessor({})
    
    def test_interruptible_without_method_raises_error(self):
        """Test that @interruptible without _handle_interrupt raises error."""
        @interruptible(save_point_id="test")
        class BadProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            # Missing _handle_interrupt method
        
        with pytest.raises(NotImplementedError):
            BadProcessor({})


class TestDecoratorDefaults:
    """Test decorator default values."""
    
    def test_observable_with_single_observer(self):
        """Test @observable with single observer."""
        @observable(observers=["default_observer"])
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        config = processor.get_capability_config('observable')
        assert config['observers'] == ["default_observer"]
    
    def test_evaluable_defaults(self):
        """Test @evaluable with default parameters."""
        @evaluable()
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        config = processor.get_capability_config('evaluable')
        assert config['max_retries'] == 3
    
    def test_debuggable_defaults(self):
        """Test @debuggable with default parameters."""
        @debuggable()
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        config = processor.get_capability_config('debuggable')
        assert config['max_retries'] == 3
    
    def test_interruptible_defaults(self):
        """Test @interruptible with default parameters."""
        @interruptible()
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "default_save_point"
    
    def test_interruptible_with_valid_save_point_id(self):
        """Test @interruptible with valid save_point_id."""
        @interruptible(save_point_id="valid_save_point")
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        config = processor.get_capability_config('interruptible')
        assert config['save_point_id'] == "valid_save_point"


class TestDecoratorInheritance:
    """Test decorator inheritance behavior."""
    
    def test_capability_inheritance(self):
        """Test that capabilities are properly inherited."""
        @observable(observers=["base"])
        @evaluable(max_retries=2)
        class BaseProcessorWithCapabilities(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        @debuggable(max_retries=1)
        class ChildProcessor(BaseProcessorWithCapabilities):
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
        
        child = ChildProcessor({})
        capabilities = child.get_capabilities()
        
        # Should have all capabilities from both base and child
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities  
        assert 'debuggable' in capabilities
        
        # Check configurations
        assert child.get_capability_config('observable')['observers'] == ["base"]
        assert child.get_capability_config('evaluable')['max_retries'] == 2
        assert child.get_capability_config('debuggable')['max_retries'] == 1


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_observable_empty_observers_raises_error(self):
        """Test @observable with empty observers list raises error."""
        with pytest.raises(ValueError, match="Observable decorator requires at least one observer"):
            @observable(observers=[])
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
    
    def test_observable_invalid_observer_type_raises_error(self):
        """Test @observable with non-string observer raises error."""
        with pytest.raises(ValueError, match="All observers must be strings"):
            @observable(observers=["valid", 123])
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
    
    def test_zero_retries(self):
        """Test decorators with zero retries."""
        @evaluable(max_retries=0)
        @debuggable(max_retries=0)
        class TestProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = TestProcessor({})
        
        assert processor.get_capability_config('evaluable')['max_retries'] == 0
        assert processor.get_capability_config('debuggable')['max_retries'] == 0
    
    def test_negative_retries_raises_error(self):
        """Test decorators with negative retries raise error."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            @evaluable(max_retries=-1)
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                
                def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                    return state
    
    def test_interruptible_empty_save_point_id_raises_error(self):
        """Test @interruptible with empty save_point_id raises error."""
        with pytest.raises(ValueError, match="save_point_id must be a non-empty string"):
            @interruptible(save_point_id="")
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                
                def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                    return state
    
    def test_interruptible_non_string_save_point_id_raises_error(self):
        """Test @interruptible with non-string save_point_id raises error."""
        with pytest.raises(ValueError, match="save_point_id must be a non-empty string"):
            @interruptible(save_point_id=123)
            class TestProcessor(BaseProcessor):
                def get_processor_type(self) -> ProcessorType:
                    return ProcessorType.DATA_FETCHER
                
                def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                    return state
                
                def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                    return state 