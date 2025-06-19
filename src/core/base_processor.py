"""
Base processor class with conditional abstract methods based on decorator capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass
from langgraph.graph import StateGraph


class ProcessorType(Enum):
    """Types of processors in the system."""
    DATA_FETCHER = "data_fetcher"
    FEATURE_BUILDER = "feature_builder"
    FACTOR_AUGMENTER = "factor_augmenter"
    BACKTEST_RUNNER = "backtest_runner"


class ProcessorState(TypedDict):
    """Minimal LangGraph state for processors."""
    input_data: Any                    # Input to be processed
    output_data: Optional[Any]         # Processing results
    error: Optional[Exception]         # Any processing errors
    status: str                        # Current processing status
    retry_count: int                   # Number of retries attempted
    interrupt_requested: bool          # UI interrupt signal
    save_point_id: Optional[str]       # Save point identifier


@dataclass
class ProcessorResult:
    """Result of processor execution."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[Exception] = None


class BaseProcessor(ABC):
    """
    Base processor class with minimal abstractions and LangGraph integration.
    
    Uses capability-based decorators to mark processor abilities:
    - @observable: Monitoring capability
    - @evaluable: Evaluation capability (requires _evaluate_result implementation)
    - @debuggable: Debug capability (requires _debug_error implementation)
    - @interruptible: UI interrupt capability (requires _handle_interrupt implementation)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration."""
        self.config = config
        self._capabilities: List[str] = []
        self._capability_configs: Dict[str, Dict[str, Any]] = {}
        
        # Extract capabilities from class decorators
        self._extract_capabilities()
        
        # Validate that required methods are implemented for capabilities
        self._validate_capability_methods()
    
    # ================================
    # CORE ABSTRACT METHODS (Always Required)
    # ================================
    
    @abstractmethod
    def get_processor_type(self) -> ProcessorType:
        """Return the type of this processor."""
        pass
    
    @abstractmethod
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """
        Core business logic of the processor.
        
        Args:
            state: Current processor state
            
        Returns:
            Updated processor state
            
        Raises:
            Exception: Any processing errors (will be handled by debug capability if available)
        """
        pass
    
    # ================================
    # CONDITIONAL ABSTRACT METHODS (Only Required if Decorated)
    # ================================
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """
        Evaluate the processing result (required if @evaluable is used).
        
        Args:
            state: Current processor state with output_data
            
        Returns:
            Updated state with eval_passed boolean set
        """
        if self.has_capability('evaluable'):
            raise NotImplementedError(
                f"{self.__class__.__name__} uses @evaluable but doesn't implement _evaluate_result"
            )
        return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """
        Debug processing errors (required if @debuggable is used).
        
        Args:
            state: Current processor state with error set
            
        Returns:
            Updated state with should_retry boolean set
        """
        if self.has_capability('debuggable'):
            raise NotImplementedError(
                f"{self.__class__.__name__} uses @debuggable but doesn't implement _debug_error"
            )
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """
        Handle UI interrupt requests (required if @interruptible is used).
        
        Args:
            state: Current processor state with interrupt_requested=True
            
        Returns:
            Updated state with interrupt handling
        """
        if self.has_capability('interruptible'):
            raise NotImplementedError(
                f"{self.__class__.__name__} uses @interruptible but doesn't implement _handle_interrupt"
            )
        return state
    
    # ================================
    # TEMPLATE METHODS (Implemented in Base)
    # ================================
    
    def create_subgraph(self) -> StateGraph:
        """
        Generate LangGraph subgraph based on processor capabilities.
        
        Returns:
            Complete StateGraph that can be used as a node in larger workflows
        """
        from .subgraph_builder import SubgraphBuilder
        builder = SubgraphBuilder()
        return builder.create_subgraph(self)
    
    def process(self, input_data: Any) -> ProcessorResult:
        """
        Execute the processor with input data.
        
        Args:
            input_data: Data to process
            
        Returns:
            ProcessorResult with success status and output
        """
        try:
            # Create initial state
            initial_state = self._initialize_state(input_data)
            
            # Get subgraph and execute
            subgraph = self.create_subgraph()
            result_state = subgraph.invoke(initial_state)
            
            # Convert to result
            return ProcessorResult(
                success=result_state.get("status") == "success",
                data=result_state.get("output_data"),
                metadata={
                    "processor_type": self.get_processor_type().value,
                    "capabilities": self.get_capabilities(),
                    "retry_count": result_state.get("retry_count", 0)
                },
                error=result_state.get("error")
            )
            
        except Exception as e:
            return ProcessorResult(
                success=False,
                data=None,
                metadata={
                    "processor_type": self.get_processor_type().value,
                    "capabilities": self.get_capabilities()
                },
                error=e
            )
    
    def get_capabilities(self) -> List[str]:
        """Return list of processor capabilities from applied decorators."""
        return self._capabilities.copy()
    
    def has_capability(self, capability: str) -> bool:
        """Check if processor has a specific capability."""
        return capability in self._capabilities
    
    def get_capability_config(self, capability: str) -> Dict[str, Any]:
        """Get configuration for a specific capability."""
        return self._capability_configs.get(capability, {})
    
    # ================================
    # PRIVATE METHODS
    # ================================
    
    def _initialize_state(self, input_data: Any) -> ProcessorState:
        """Initialize processor state for execution."""
        return ProcessorState(
            input_data=input_data,
            output_data=None,
            error=None,
            status="initialized",
            retry_count=0,
            interrupt_requested=False,
            save_point_id=None
        )
    
    def _extract_capabilities(self) -> None:
        """Extract capabilities from class-level decorators."""
        # Check for capability markers set by decorators
        if hasattr(self.__class__, '_processor_capabilities'):
            self._capabilities = getattr(self.__class__, '_processor_capabilities', [])
        
        if hasattr(self.__class__, '_processor_capability_configs'):
            self._capability_configs = getattr(self.__class__, '_processor_capability_configs', {})
    
    def _validate_capability_methods(self) -> None:
        """Validate that required methods are implemented for each capability."""
        if self.has_capability('evaluable'):
            # Check if _evaluate_result is properly implemented
            method = getattr(self.__class__, '_evaluate_result', None)
            if method is None or method is BaseProcessor._evaluate_result:
                raise NotImplementedError(
                    f"{self.__class__.__name__} uses @evaluable but doesn't implement _evaluate_result"
                )
        
        if self.has_capability('debuggable'):
            # Check if _debug_error is properly implemented
            method = getattr(self.__class__, '_debug_error', None)
            if method is None or method is BaseProcessor._debug_error:
                raise NotImplementedError(
                    f"{self.__class__.__name__} uses @debuggable but doesn't implement _debug_error"
                )
        
        if self.has_capability('interruptible'):
            # Check if _handle_interrupt is properly implemented
            method = getattr(self.__class__, '_handle_interrupt', None)
            if method is None or method is BaseProcessor._handle_interrupt:
                raise NotImplementedError(
                    f"{self.__class__.__name__} uses @interruptible but doesn't implement _handle_interrupt"
                ) 