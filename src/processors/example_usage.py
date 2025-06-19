"""
Example usage of refactored processors with new architecture.

This example demonstrates:
1. Creating processors with capability decorators
2. Using ProcessorFactory for processor management
3. Generating dynamic LangGraph subgraphs
4. Running complete pipeline with PipelineCoordinator
"""

import logging
from typing import Dict, Any

from ..core.processor_factory import ProcessorFactory
from ..core.pipeline_coordinator import PipelineCoordinator
from ..core.base_processor import ProcessorType
from . import DataFetcher, FeatureBuilder, FactorAugmenter, BacktestRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_config() -> Dict[str, Any]:
    """Create mock configuration for demonstration."""
    return {
        'model_name': 'gpt-4',
        'prompt_manager': MockPromptManager(),
        'db_path': 'mock://kdb',
        'backtest_engine': MockBacktestEngine(),
        'min_rows_per_table': 50,
        'min_features_count': 2,
        'min_sharpe_ratio': 0.5,
        'max_drawdown_threshold': -0.3
    }


class MockPromptManager:
    """Mock prompt manager for demonstration."""
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """Mock prompt generation."""
        return f"Generate {template_name} with parameters: {kwargs}"


class MockBacktestEngine:
    """Mock backtest engine for demonstration."""
    
    def pause(self):
        """Mock pause functionality."""
        pass


def demonstrate_individual_processor():
    """Demonstrate using a single processor."""
    logger.info("=== Individual Processor Demonstration ===")
    
    # Create configuration
    config = create_mock_config()
    
    # Create DataFetcher with all capabilities
    data_fetcher = DataFetcher(config)
    
    # Show processor capabilities
    capabilities = data_fetcher.get_capabilities()
    logger.info(f"DataFetcher capabilities: {capabilities}")
    
    # Create input data
    input_data = {
        'feature_description': 'Get daily price data for AAPL stock'
    }
    
    # Process data using the processor
    result = data_fetcher.process(input_data)
    
    logger.info(f"Processing result: success={result.success}")
    logger.info(f"Result metadata: {result.metadata}")
    
    if result.success:
        logger.info("✅ Data fetching completed successfully")
    else:
        logger.error(f"❌ Data fetching failed: {result.error}")


def demonstrate_processor_factory():
    """Demonstrate using ProcessorFactory."""
    logger.info("=== Processor Factory Demonstration ===")
    
    # Create factory
    factory = ProcessorFactory()
    
    # Register processor classes
    factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
    factory.register_processor_class(ProcessorType.FEATURE_BUILDER, FeatureBuilder)
    factory.register_processor_class(ProcessorType.FACTOR_AUGMENTER, FactorAugmenter)
    factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, BacktestRunner)
    
    # Show factory status
    status = factory.get_factory_status()
    logger.info(f"Factory status: {status}")
    
    # Create processor using factory
    config = create_mock_config()
    data_fetcher = factory.create_processor(ProcessorType.DATA_FETCHER, config)
    
    # Generate subgraph for processor
    subgraph = factory.create_processor_subgraph(data_fetcher)
    logger.info(f"✅ Generated subgraph for DataFetcher: {type(subgraph)}")
    
    # Validate processor capabilities
    is_valid = factory.validate_processor_capabilities(data_fetcher)
    logger.info(f"Processor validation: {is_valid}")


def demonstrate_pipeline_coordination():
    """Demonstrate using PipelineCoordinator."""
    logger.info("=== Pipeline Coordination Demonstration ===")
    
    # Create factory and register processors
    factory = ProcessorFactory()
    factory.register_processor_class(ProcessorType.DATA_FETCHER, DataFetcher)
    factory.register_processor_class(ProcessorType.FEATURE_BUILDER, FeatureBuilder)
    factory.register_processor_class(ProcessorType.FACTOR_AUGMENTER, FactorAugmenter)
    factory.register_processor_class(ProcessorType.BACKTEST_RUNNER, BacktestRunner)
    
    # Create processors
    config = create_mock_config()
    processors = [
        factory.create_processor(ProcessorType.DATA_FETCHER, config),
        factory.create_processor(ProcessorType.FEATURE_BUILDER, config),
        factory.create_processor(ProcessorType.FACTOR_AUGMENTER, config),
        factory.create_processor(ProcessorType.BACKTEST_RUNNER, config)
    ]
    
    # Create pipeline coordinator
    coordinator = PipelineCoordinator()
    
    # Build pipeline workflow
    pipeline = coordinator.build_pipeline_workflow(processors)
    logger.info(f"✅ Built pipeline with {len(processors)} processors")
    
    # Show pipeline status
    status = coordinator.get_pipeline_status()
    logger.info(f"Pipeline status: {status}")
    
    # Execute pipeline (mock)
    initial_data = {
        'feature_description': 'Build momentum factors for AAPL',
        'strategy_spec': {'type': 'momentum', 'lookback': 20}
    }
    
    try:
        # In real usage, you would execute the pipeline:
        # result = coordinator.execute_pipeline(initial_data)
        logger.info("✅ Pipeline ready for execution")
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")


def demonstrate_capability_system():
    """Demonstrate the capability-based decorator system."""
    logger.info("=== Capability System Demonstration ===")
    
    config = create_mock_config()
    
    # Create processors with different capabilities
    processors_info = [
        (DataFetcher(config), "Data fetching with observability, evaluation, debugging, and interrupts"),
        (FeatureBuilder(config), "Feature building with observability, evaluation, and debugging"),
        (FactorAugmenter(config), "Factor augmentation with observability, evaluation, and debugging"),
        (BacktestRunner(config), "Backtesting with all capabilities including interrupts")
    ]
    
    for processor, description in processors_info:
        capabilities = processor.get_capabilities()
        processor_type = processor.get_processor_type()
        
        logger.info(f"\n{processor_type.value.upper()}:")
        logger.info(f"  Description: {description}")
        logger.info(f"  Capabilities: {capabilities}")
        
        # Show capability configurations
        for capability in capabilities:
            config_info = processor.get_capability_config(capability)
            logger.info(f"  {capability} config: {config_info}")


def main():
    """Run all demonstrations."""
    logger.info("🚀 Starting Processor Architecture Demonstration")
    
    try:
        demonstrate_individual_processor()
        print("\n" + "="*60 + "\n")
        
        demonstrate_processor_factory()
        print("\n" + "="*60 + "\n")
        
        demonstrate_capability_system()
        print("\n" + "="*60 + "\n")
        
        demonstrate_pipeline_coordination()
        
        logger.info("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 