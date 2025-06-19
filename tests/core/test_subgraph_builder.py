"""
Tests for src/core/subgraph_builder.py

Tests dynamic LangGraph subgraph generation based on processor capabilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.subgraph_builder import SubgraphBuilder
from src.core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from src.core.decorators import observable, evaluable, debuggable, interruptible


class TestSubgraphBuilder:
    """Test SubgraphBuilder functionality."""
    
    def test_basic_subgraph_creation(self):
        """Test basic subgraph creation for processor without capabilities."""
        class BasicProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = BasicProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        # Basic validation
        assert subgraph is not None
        # SubgraphBuilder returns CompiledGraph
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_interruptible_subgraph(self):
        """Test subgraph creation for interruptible processor."""
        @interruptible(save_point_id="test")
        class InterruptibleProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = InterruptibleProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        assert subgraph is not None
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_evaluable_subgraph(self):
        """Test subgraph creation for evaluable processor."""
        @evaluable(max_retries=2)
        class EvaluableProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FEATURE_BUILDER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = EvaluableProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        assert subgraph is not None
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_debuggable_subgraph(self):
        """Test subgraph creation for debuggable processor."""
        @debuggable(max_retries=3)
        class DebuggableProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.FACTOR_AUGMENTER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = DebuggableProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        assert subgraph is not None
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_complex_subgraph_multiple_capabilities(self):
        """Test subgraph creation for processor with multiple capabilities."""
        @observable(observers=["test_observer"])
        @evaluable(max_retries=2)
        @debuggable(max_retries=3)
        @interruptible(save_point_id="complex_test")
        class ComplexProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.BACKTEST_RUNNER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _debug_error(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = ComplexProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        assert subgraph is not None
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_observable_subgraph(self):
        """Test subgraph creation for observable processor."""
        @observable(observers=["test_observer"])
        class ObservableProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
        
        processor = ObservableProcessor({})
        builder = SubgraphBuilder()
        
        subgraph = builder.create_subgraph(processor)
        
        assert subgraph is not None
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)
    
    def test_subgraph_error_handling(self):
        """Test subgraph creation error handling."""
        class ProblematicProcessor(BaseProcessor):
            def get_processor_type(self) -> ProcessorType:
                return ProcessorType.DATA_FETCHER
            
            def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
                return state
            
            def get_capabilities(self):
                # Return invalid capabilities to test error handling
                return ["nonexistent_capability"]
        
        processor = ProblematicProcessor({})
        builder = SubgraphBuilder()
        
        # Should not raise error - builder handles gracefully
        subgraph = builder.create_subgraph(processor)
        assert subgraph is not None
