"""
Integration tests for processors.

Tests the interaction between different processors and their integration with the AI service.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.processors.data_fetcher import DataFetcher
from src.processors.feature_builder import FeatureBuilder
from src.processors.factor_augmenter import FactorAugmenter
from src.processors.backtest_runner import BacktestRunner
from src.core.base_processor import ProcessorType, ProcessorState
from src.core.processor_factory import ProcessorFactory


class TestProcessorsIntegration:
    """Test integration between different processors."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        
        # Common configuration for all processors
        self.base_config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager
        }
        
        # Specific configurations for each processor
        self.data_fetcher_config = {
            **self.base_config,
            'db_path': '/test/db',
            'min_rows_per_table': 50
        }
        
        self.feature_builder_config = {
            **self.base_config,
            'feature_template': 'test_template'
        }
        
        self.factor_augmenter_config = {
            **self.base_config,
            'augmentation_methods': ['pca', 'ica']
        }
        
        self.backtest_runner_config = {
            **self.base_config,
            'strategy_template': 'test_strategy'
        }
    
    def test_data_fetcher_to_feature_builder(self):
        """Test data flow from DataFetcher to FeatureBuilder."""
        # Initialize processors
        fetcher = DataFetcher(self.data_fetcher_config)
        builder = FeatureBuilder(self.feature_builder_config)
        
        # Mock LLM responses
        self.mock_llm.invoke.side_effect = [
            Mock(content='{"query": "select from trades"}'),
            Mock(content='{"feature": "test_feature"}')
        ]
        
        # Create initial state
        state: ProcessorState = {
            'input_data': {'feature_description': 'Test feature'},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        # Process through DataFetcher
        fetcher_state = fetcher._process_core_logic(state)
        assert fetcher_state['status'] == 'success'
        
        # Process through FeatureBuilder
        builder_state = builder._process_core_logic(fetcher_state)
        assert builder_state['status'] == 'success'


class TestProcessorsArchitectureCompatibility:
    """Test all processors for new architecture compatibility."""
    
    def setup_method(self):
        """Setup common test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        self.mock_backtest_engine = Mock()
    
    def test_data_fetcher_architecture(self):
        """Test DataFetcher architecture compatibility."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'db_path': '/test/db'
        }
        
        fetcher = DataFetcher(config)
        
        # Test basic properties
        assert fetcher.get_processor_type() == ProcessorType.DATA_FETCHER
        
        # Test capabilities
        capabilities = fetcher.get_capabilities()
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
        
        # Test required methods exist
        assert hasattr(fetcher, '_process_core_logic')
        assert hasattr(fetcher, '_evaluate_result')
        assert hasattr(fetcher, '_debug_error')
        assert hasattr(fetcher, '_handle_interrupt')
        
        # Test method signatures
        assert callable(getattr(fetcher, '_process_core_logic'))
        assert callable(getattr(fetcher, '_evaluate_result'))
        assert callable(getattr(fetcher, '_debug_error'))
        assert callable(getattr(fetcher, '_handle_interrupt'))
    
    def test_feature_builder_architecture(self):
        """Test FeatureBuilder architecture compatibility."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'test_data': {}
        }
        
        builder = FeatureBuilder(config)
        
        # Test basic properties
        assert builder.get_processor_type() == ProcessorType.FEATURE_BUILDER
        
        # Test capabilities
        capabilities = builder.get_capabilities()
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' not in capabilities  # FeatureBuilder is not interruptible
        
        # Test required methods exist
        assert hasattr(builder, '_process_core_logic')
        assert hasattr(builder, '_evaluate_result')
        assert hasattr(builder, '_debug_error')
        assert not hasattr(builder, '_handle_interrupt')  # Should not have this method
    
    def test_factor_augmenter_architecture(self):
        """Test FactorAugmenter architecture compatibility."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'enhancement_methods': ['transform', 'combine']
        }
        
        augmenter = FactorAugmenter(config)
        
        # Test basic properties
        assert augmenter.get_processor_type() == ProcessorType.FACTOR_AUGMENTER
        
        # Test capabilities
        capabilities = augmenter.get_capabilities()
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' not in capabilities  # FactorAugmenter is not interruptible
        
        # Test required methods exist
        assert hasattr(augmenter, '_process_core_logic')
        assert hasattr(augmenter, '_evaluate_result')
        assert hasattr(augmenter, '_debug_error')
    
    def test_backtest_runner_architecture(self):
        """Test BacktestRunner architecture compatibility."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'backtest_engine': self.mock_backtest_engine,
            'performance_metrics': ['sharpe_ratio']
        }
        
        runner = BacktestRunner(config)
        
        # Test basic properties
        assert runner.get_processor_type() == ProcessorType.BACKTEST_RUNNER
        
        # Test capabilities
        capabilities = runner.get_capabilities()
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
        
        # Test required methods exist
        assert hasattr(runner, '_process_core_logic')
        assert hasattr(runner, '_evaluate_result')
        assert hasattr(runner, '_debug_error')
        assert hasattr(runner, '_handle_interrupt')


class TestProcessorFactoryIntegration:
    """Test processor factory integration with all processors."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.factory = ProcessorFactory()
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        self.mock_backtest_engine = Mock()
    
    def test_register_all_processors(self):
        """Test registering all processor types."""
        # Register all processors
        self.factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
        self.factory.register_processor_class(ProcessorType.FEATURE_BUILDER, FeatureBuilder)
        self.factory.register_processor_class(ProcessorType.FACTOR_AUGMENTER, FactorAugmenter)
        self.factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, BacktestRunner)
        
        # Verify all are registered
        available_types = self.factory.get_available_processors()
        assert ProcessorType.DATA_FETCHER in available_types
        assert ProcessorType.FEATURE_BUILDER in available_types
        assert ProcessorType.FACTOR_AUGMENTER in available_types
        assert ProcessorType.BACKTEST_RUNNER in available_types
        assert len(available_types) == 4
    
    def test_create_all_processor_types(self):
        """Test creating all processor types through factory."""
        # Register processors
        self.factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
        self.factory.register_processor_class(ProcessorType.FEATURE_BUILDER, FeatureBuilder)
        self.factory.register_processor_class(ProcessorType.FACTOR_AUGMENTER, FactorAugmenter)
        self.factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, BacktestRunner)
        
        # Create each processor type
        configs = {
            ProcessorType.DATA_FETCHER: {
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'db_path': '/test'
            },
            ProcessorType.FEATURE_BUILDER: {
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'test_data': {}
            },
            ProcessorType.FACTOR_AUGMENTER: {
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'enhancement_methods': ['transform']
            },
            ProcessorType.BACKTEST_RUNNER: {
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'backtest_engine': self.mock_backtest_engine
            }
        }
        
        created_processors = {}
        for processor_type, config in configs.items():
            processor = self.factory.create_processor(processor_type, config)
            created_processors[processor_type] = processor
            
            # Verify processor type
            assert processor.get_processor_type() == processor_type
            
            # Verify capabilities are detected
            capabilities = self.factory.get_processor_capabilities(processor_type)
            assert len(capabilities) > 0
    
    def test_processor_capabilities_detection(self):
        """Test capability detection for all processors."""
        # Register processors
        self.factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
        self.factory.register_processor_class(ProcessorType.FEATURE_BUILDER, FeatureBuilder)
        self.factory.register_processor_class(ProcessorType.FACTOR_AUGMENTER, FactorAugmenter)
        self.factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, BacktestRunner)
        
        # Test DataFetcher capabilities
        data_fetcher_caps = self.factory.get_processor_capabilities(ProcessorType.DATA_FETCHER)
        assert 'observable' in data_fetcher_caps
        assert 'evaluable' in data_fetcher_caps
        assert 'debuggable' in data_fetcher_caps
        assert 'interruptible' in data_fetcher_caps
        
        # Test FeatureBuilder capabilities
        feature_builder_caps = self.factory.get_processor_capabilities(ProcessorType.FEATURE_BUILDER)
        assert 'observable' in feature_builder_caps
        assert 'evaluable' in feature_builder_caps
        assert 'debuggable' in feature_builder_caps
        assert 'interruptible' not in feature_builder_caps
        
        # Test FactorAugmenter capabilities
        factor_augmenter_caps = self.factory.get_processor_capabilities(ProcessorType.FACTOR_AUGMENTER)
        assert 'observable' in factor_augmenter_caps
        assert 'evaluable' in factor_augmenter_caps
        assert 'debuggable' in factor_augmenter_caps
        assert 'interruptible' not in factor_augmenter_caps
        
        # Test BacktestRunner capabilities
        backtest_runner_caps = self.factory.get_processor_capabilities(ProcessorType.BACKTEST_RUNNER)
        assert 'observable' in backtest_runner_caps
        assert 'evaluable' in backtest_runner_caps
        assert 'debuggable' in backtest_runner_caps
        assert 'interruptible' in backtest_runner_caps


class TestProcessorStateCompatibility:
    """Test processor state handling compatibility."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        
        self.sample_state: ProcessorState = {
            'input_data': {},
            'output_data': {},
            'status': 'pending',
            'metadata': {}
        }
    
    def test_data_fetcher_state_handling(self):
        """Test DataFetcher state handling."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'db_path': '/test'
        }
        
        fetcher = DataFetcher(config)
        
        # Test state with error
        error_state = self.sample_state.copy()
        error_state['error'] = Exception("Test error")
        
        debug_result = fetcher._debug_error(error_state)
        assert 'should_retry' in debug_result
        assert 'debug_reason' in debug_result
        
        # Test interrupt handling
        interrupt_state = self.sample_state.copy()
        interrupt_state['status'] = 'running'
        
        interrupt_result = fetcher._handle_interrupt(interrupt_state)
        assert interrupt_result['status'] == 'paused'
    
    def test_feature_builder_state_handling(self):
        """Test FeatureBuilder state handling."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'test_data': {}
        }
        
        builder = FeatureBuilder(config)
        
        # Test evaluation state
        eval_state = self.sample_state.copy()
        eval_state['output_data'] = {}
        
        eval_result = builder._evaluate_result(eval_state)
        assert 'eval_passed' in eval_result
        assert 'eval_reason' in eval_result
    
    def test_factor_augmenter_state_handling(self):
        """Test FactorAugmenter state handling."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'enhancement_methods': ['transform']
        }
        
        augmenter = FactorAugmenter(config)
        
        # Test evaluation with good data
        eval_state = self.sample_state.copy()
        eval_state['output_data'] = {
            'enhanced_factors': {'factor1': {'values': [1, 2, 3]}},
            'raw_features_count': 1,
            'enhancement_strategies': [{'method': 'transform'}]
        }
        
        with pytest.mock.patch.object(augmenter, '_validate_factor_quality', return_value=True):
            eval_result = augmenter._evaluate_result(eval_state)
            assert 'eval_passed' in eval_result
    
    def test_backtest_runner_state_handling(self):
        """Test BacktestRunner state handling."""
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'backtest_engine': Mock()
        }
        
        runner = BacktestRunner(config)
        
        # Test evaluation with performance metrics
        eval_state = self.sample_state.copy()
        eval_state['output_data'] = {
            'performance_metrics': {
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.1,
                'total_return': 0.15
            },
            'backtest_results': {'trades_count': 20}
        }
        
        with pytest.mock.patch.object(runner, '_validate_strategy_stability', return_value=True):
            eval_result = runner._evaluate_result(eval_state)
            assert 'eval_passed' in eval_result


class TestProcessorWorkflowCompatibility:
    """Test processor compatibility with workflow systems."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        self.mock_backtest_engine = Mock()
    
    def test_processor_subgraph_creation(self):
        """Test that all processors can have subgraphs created."""
        from src.core.subgraph_builder import SubgraphBuilder
        
        processors = [
            DataFetcher({
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'db_path': '/test'
            }),
            FeatureBuilder({
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'test_data': {}
            }),
            FactorAugmenter({
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'enhancement_methods': ['transform']
            }),
            BacktestRunner({
                'model_name': 'gpt-4',
                'prompt_manager': self.mock_prompt_manager,
                'backtest_engine': self.mock_backtest_engine
            })
        ]
        
        builder = SubgraphBuilder()
        
        for processor in processors:
            # Should not raise any exceptions
            subgraph = builder.create_subgraph(processor)
            assert subgraph is not None
            
            # Verify it's a compiled graph
            from langgraph.graph.graph import CompiledGraph
            assert isinstance(subgraph, CompiledGraph)
    
    def test_processor_factory_subgraph_integration(self):
        """Test processor factory subgraph creation."""
        factory = ProcessorFactory()
        
        # Register and create a processor
        factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
        
        config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'db_path': '/test'
        }
        
        processor = factory.create_processor(ProcessorType.DATA_FETCHER, config)
        
        # Create subgraph through factory
        subgraph = factory.create_processor_subgraph(processor)
        assert subgraph is not None
        
        from langgraph.graph.graph import CompiledGraph
        assert isinstance(subgraph, CompiledGraph)


def test_all_processors_summary():
    """Summary test to verify all processors are working with new architecture."""
    print("\n=== Processor Architecture Compatibility Summary ===")
    
    mock_llm = Mock()
    mock_prompt_manager = Mock()
    mock_backtest_engine = Mock()
    
    processors_info = [
        (DataFetcher, ProcessorType.DATA_FETCHER, {
            'model_name': 'gpt-4',
            'prompt_manager': mock_prompt_manager,
            'db_path': '/test'
        }),
        (FeatureBuilder, ProcessorType.FEATURE_BUILDER, {
            'model_name': 'gpt-4',
            'prompt_manager': mock_prompt_manager,
            'test_data': {}
        }),
        (FactorAugmenter, ProcessorType.FACTOR_AUGMENTER, {
            'model_name': 'gpt-4',
            'prompt_manager': mock_prompt_manager,
            'enhancement_methods': ['transform']
        }),
        (BacktestRunner, ProcessorType.BACKTEST_RUNNER, {
            'model_name': 'gpt-4',
            'prompt_manager': mock_prompt_manager,
            'backtest_engine': mock_backtest_engine
        })
    ]
    
    all_compatible = True
    
    for processor_class, processor_type, config in processors_info:
        try:
            processor = processor_class(config)
            capabilities = processor.get_capabilities()
            
            print(f"✅ {processor_class.__name__}")
            print(f"   Type: {processor_type.value}")
            print(f"   Capabilities: {', '.join(capabilities)}")
            
            # Verify required methods
            required_methods = ['_process_core_logic']
            if 'evaluable' in capabilities:
                required_methods.append('_evaluate_result')
            if 'debuggable' in capabilities:
                required_methods.append('_debug_error')
            if 'interruptible' in capabilities:
                required_methods.append('_handle_interrupt')
            
            missing_methods = [m for m in required_methods if not hasattr(processor, m)]
            if missing_methods:
                print(f"   ❌ Missing methods: {', '.join(missing_methods)}")
                all_compatible = False
            else:
                print(f"   ✅ All required methods present")
            
        except Exception as e:
            print(f"❌ {processor_class.__name__}: {e}")
            all_compatible = False
        
        print()
    
    print("=== Summary ===")
    if all_compatible:
        print("✅ All processors are compatible with new decorator-based architecture!")
    else:
        print("❌ Some processors need updates for new architecture compatibility.")
    
    assert all_compatible, "Not all processors are compatible with new architecture" 