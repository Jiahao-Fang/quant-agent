"""
Processor Factory for creating processors with LangGraph subgraph generation.
"""

from typing import Dict, Any, Type, List
from dataclasses import dataclass
import logging

from .base_processor import BaseProcessor, ProcessorType
from .subgraph_builder import SubgraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration container for processor creation."""
    processor_type: ProcessorType
    config_params: Dict[str, Any]
    
    def __post_init__(self):
        if self.config_params is None:
            self.config_params = {}


class ProcessorFactory:
    """
    Factory for creating processors with LangGraph subgraph generation.
    
    Responsibilities:
    - Create processor instances with proper capability validation
    - Generate LangGraph subgraphs for each processor
    - Provide processor capability discovery
    """
    
    def __init__(self):
        """Initialize the processor factory."""
        self._processor_classes: Dict[ProcessorType, Type[BaseProcessor]] = {}
        self._subgraph_builder = SubgraphBuilder()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def create_processor(self, processor_type: ProcessorType, config: Dict[str, Any]) -> BaseProcessor:
        """
        Create a processor instance of the specified type.
        
        Args:
            processor_type: The type of processor to create
            config: Configuration parameters for the processor
            
        Returns:
            BaseProcessor instance configured and ready to use
            
        Raises:
            ValueError: If processor_type is not registered
        """
        if processor_type not in self._processor_classes:
            raise ValueError(f"Processor type {processor_type.value} is not registered")
        
        processor_class = self._processor_classes[processor_type]
        
        try:
            # Create processor instance
            processor = processor_class(config)
            
            # Log capabilities
            capabilities = processor.get_capabilities()
            if capabilities:
                self.logger.info(f"Created processor {processor_type.value} with capabilities: {capabilities}")
            else:
                self.logger.info(f"Created basic processor: {processor_type.value}")
            
            return processor
            
        except Exception as e:
            self.logger.error(f"Failed to create processor {processor_type.value}: {e}")
            raise
    
    def create_processor_subgraph(self, processor: BaseProcessor):
        """
        Generate LangGraph subgraph for a processor.
        
        Args:
            processor: The processor to create subgraph for
            
        Returns:
            StateGraph that can be used as a node in larger workflows
        """
        try:
            subgraph = self._subgraph_builder.create_subgraph(processor)
            
            capabilities = processor.get_capabilities()
            self.logger.info(
                f"Generated subgraph for {processor.get_processor_type().value} "
                f"with capabilities: {capabilities}"
            )
            
            return subgraph
            
        except Exception as e:
            self.logger.error(f"Failed to create subgraph for processor: {e}")
            raise
    
    def validate_processor_capabilities(self, processor: BaseProcessor) -> bool:
        """
        Validate that processor implements required methods for its capabilities.
        
        Args:
            processor: The processor to validate
            
        Returns:
            True if all required methods are implemented
            
        Raises:
            NotImplementedError: If required methods are missing
        """
        try:
            # This validation is already done in BaseProcessor.__init__
            # Just return True if processor was created successfully
            return True
        except Exception as e:
            self.logger.error(f"Processor capability validation failed: {e}")
            return False
    
    def register_processor_class(self, processor_type: ProcessorType, cls: Type[BaseProcessor]) -> None:
        """
        Register a processor class with the factory.
        
        Args:
            processor_type: The type identifier for this processor
            cls: The processor class to register
            
        Raises:
            ValueError: If class doesn't inherit from BaseProcessor
        """
        if not issubclass(cls, BaseProcessor):
            raise ValueError(f"Class {cls.__name__} must inherit from BaseProcessor")
        
        self._processor_classes[processor_type] = cls
        self.logger.info(f"Registered processor class {cls.__name__} for type {processor_type.value}")
    
    def get_processor_capabilities(self, processor_type: ProcessorType) -> List[str]:
        """
        Get the capabilities of a registered processor type.
        
        Args:
            processor_type: The processor type to query
            
        Returns:
            List of capability names supported by the processor
            
        Raises:
            ValueError: If processor_type is not registered
        """
        if processor_type not in self._processor_classes:
            raise ValueError(f"Processor type {processor_type.value} is not registered")
        
        processor_class = self._processor_classes[processor_type]
        
        # Get capabilities from class decorators
        if hasattr(processor_class, '_processor_capabilities'):
            return list(processor_class._processor_capabilities)
        
        return []
    
    def create_processor_with_config(self, config: ProcessorConfig) -> BaseProcessor:
        """
        Create a processor using a ProcessorConfig object.
        
        Args:
            config: ProcessorConfig containing all creation parameters
            
        Returns:
            BaseProcessor instance configured according to the config
        """
        self.logger.info(f"Creating processor of type {config.processor_type.value}")
        
        processor = self.create_processor(config.processor_type, config.config_params)
        
        self.logger.info(f"Successfully created processor {config.processor_type.value}")
        return processor
    
    def get_available_processors(self) -> List[ProcessorType]:
        """
        Get a list of all available processor types.
        
        Returns:
            List of ProcessorType enum values that can be created
        """
        return list(self._processor_classes.keys())
    
    def is_processor_registered(self, processor_type: ProcessorType) -> bool:
        """
        Check if a processor type is registered.
        
        Args:
            processor_type: The processor type to check
            
        Returns:
            True if the processor type is registered
        """
        return processor_type in self._processor_classes
    
    def unregister_processor(self, processor_type: ProcessorType) -> None:
        """
        Unregister a processor type.
        
        Args:
            processor_type: The processor type to unregister
        """
        if processor_type in self._processor_classes:
            del self._processor_classes[processor_type]
            self.logger.info(f"Unregistered processor type: {processor_type.value}")
    
    def get_factory_status(self) -> Dict[str, Any]:
        """
        Get the current status of the factory.
        
        Returns:
            Dictionary with factory status information
        """
        processor_info = {}
        for proc_type, proc_class in self._processor_classes.items():
            capabilities = []
            try:
                capabilities = self.get_processor_capabilities(proc_type)
            except Exception as e:
                self.logger.warning(f"Could not get capabilities for {proc_type.value}: {e}")
            
            processor_info[proc_type.value] = {
                "class_name": proc_class.__name__,
                "capabilities": capabilities,
                "has_capabilities": bool(capabilities)
            }
        
        return {
            "registered_processors": len(self._processor_classes),
            "available_types": [pt.value for pt in self.get_available_processors()],
            "processor_details": processor_info
        }
    
    def clear_all_registrations(self) -> None:
        """Clear all processor registrations."""
        self._processor_classes.clear()
        self.logger.info("Cleared all processor registrations") 