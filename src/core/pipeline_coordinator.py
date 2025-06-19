"""
Pipeline Coordinator for orchestrating multi-processor workflows using LangGraph.

This module provides the PipelineCoordinator that creates and manages
LangGraph workflows composed of processor subgraphs.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .base_processor import BaseProcessor, ProcessorType, ProcessorResult, ProcessorState
from .processor_factory import ProcessorFactory

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    pipeline_id: str
    success: bool
    processor_results: List[ProcessorResult]
    final_data: Any
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get the total pipeline execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    pipeline_id: str
    enable_checkpointing: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    interrupt_on_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineCoordinator:
    """
    Pipeline coordinator that creates LangGraph workflows from processor subgraphs.
    
    Responsibilities:
    - Build main pipeline LangGraph workflow  
    - Integrate processor subgraphs as composite nodes
    - Coordinate data flow between processor subgraphs
    - Provide unified control interface (pause, resume, monitor)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline coordinator.
        
        Args:
            config: Optional global configuration for the coordinator
        """
        self.config = config or {}
        self.processors: Dict[str, BaseProcessor] = {}  # Add processors dict for management
        self.workflow: Optional[CompiledStateGraph] = None  # Add workflow attribute
        self._current_pipeline: Optional[CompiledStateGraph] = None
        self._current_config: Optional[PipelineConfig] = None
        self._execution_history: List[PipelineResult] = []
        self._status = PipelineStatus.IDLE
        self._processor_factory = ProcessorFactory()
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def add_processor(self, name: str, processor: BaseProcessor) -> None:
        """
        Add a processor to the coordinator.
        
        Args:
            name: Unique name for the processor
            processor: Processor instance to add
        """
        self.processors[name] = processor
        self.logger.info(f"Added processor: {name} ({processor.get_processor_type().value})")
    
    def remove_processor(self, name: str) -> None:
        """
        Remove a processor from the coordinator.
        
        Args:
            name: Name of the processor to remove
        """
        if name in self.processors:
            del self.processors[name]
            self.logger.info(f"Removed processor: {name}")
    
    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """
        Get a processor by name.
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor instance or None if not found
        """
        return self.processors.get(name)
    
    def create_workflow(self) -> None:
        """
        Create workflow from currently added processors.
        """
        if not self.processors:
            self.logger.warning("Creating workflow with no processors")
            # Create empty workflow
            graph = StateGraph(ProcessorState)
            graph.add_node("empty", lambda state: state)
            graph.add_edge(START, "empty")
            graph.add_edge("empty", END)
            self.workflow = graph.compile()
            return
        
        processor_list = list(self.processors.values())
        self.workflow = self.build_pipeline_workflow(processor_list)
    
    def resume_pipeline(self, resume_data: Dict[str, Any]) -> Any:
        """
        Resume a paused pipeline.
        
        Args:
            resume_data: Data containing resume information
            
        Returns:
            Resume result
        """
        if not self.workflow:
            raise RuntimeError("No workflow available for resume")
        
        self.logger.info("Resuming pipeline execution")
        return self.workflow.invoke(resume_data)
    
    def build_pipeline_workflow(self, processors: List[BaseProcessor], config: Optional[PipelineConfig] = None) -> CompiledStateGraph:
        """
        Build main pipeline LangGraph workflow from processor subgraphs.
        
        Args:
            processors: List of processor instances
            config: Optional pipeline configuration
            
        Returns:
            Compiled LangGraph StateGraph ready for execution
        """
        if not processors:
            raise ValueError("At least one processor is required")
        
        pipeline_config = config or PipelineConfig(
            pipeline_id=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        self.logger.info(f"Building pipeline workflow: {pipeline_config.pipeline_id}")
        
        # Create main pipeline state graph
        pipeline_graph = StateGraph(ProcessorState)
        
        # Add processor subgraphs as nodes
        for i, processor in enumerate(processors):
            processor_name = f"{processor.get_processor_type().value}_{i}"
            
            # Get subgraph for processor
            processor_subgraph = self._processor_factory.create_processor_subgraph(processor)
            
            # Add as node in main pipeline
            pipeline_graph.add_node(processor_name, self._create_processor_node(processor, processor_subgraph))
            
            self.logger.info(f"Added processor node: {processor_name} with capabilities: {processor.get_capabilities()}")
        
        # Connect processors in sequence
        self._connect_processors_sequentially(pipeline_graph, processors)
        
        # Add error handling and checkpointing if enabled
        if pipeline_config.enable_checkpointing:
            self._add_checkpointing_support(pipeline_graph, pipeline_config)
        
        # Compile the pipeline
        compiled_pipeline = pipeline_graph.compile()
        
        # Store current pipeline
        self._current_pipeline = compiled_pipeline
        self._current_config = pipeline_config
        
        self.logger.info(f"Successfully built pipeline with {len(processors)} processors")
        return compiled_pipeline
    
    def execute_pipeline(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the current pipeline with input data.
        
        Args:
            input_data: Initial data for the pipeline
            config: Optional execution configuration
            
        Returns:
            PipelineResult with execution outcome
        """
        # Use workflow if available, otherwise fall back to _current_pipeline
        pipeline_to_execute = self.workflow or self._current_pipeline
        
        if not pipeline_to_execute:
            # If no workflow exists, create one automatically
            if self.processors:
                self.create_workflow()
                pipeline_to_execute = self.workflow
            else:
                raise RuntimeError("No pipeline available. Add processors and call create_workflow() first.")
        
        pipeline_id = self._current_config.pipeline_id if self._current_config else "unknown"
        start_time = datetime.now()
        
        self.logger.info(f"Starting pipeline execution: {pipeline_id}")
        self._status = PipelineStatus.RUNNING
        
        try:
            # Create initial state
            initial_state = {
                'input_data': input_data,
                'output_data': None,
                'error': None,
                'status': 'starting',
                'retry_count': 0,
                'interrupt_requested': False,
                'save_point_id': None
            }
            
            # Execute pipeline
            execution_config = config or {}
            final_state = pipeline_to_execute.invoke(initial_state, config=execution_config)
            
            # Determine success
            success = final_state.get('status') == 'success' and not final_state.get('error')
            
            # Create result
            result = PipelineResult(
                pipeline_id=pipeline_id,
                success=success,
                processor_results=[],  # Would be populated in real implementation
                final_data=final_state.get('output_data'),
                start_time=start_time,
                end_time=datetime.now(),
                metadata={
                    'final_state': final_state,
                    'execution_config': execution_config
                }
            )
            
            self._execution_history.append(result)
            self._status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
            
            self.logger.info(f"Pipeline execution completed: {pipeline_id} (Success: {success})")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self._status = PipelineStatus.FAILED
            
            # Create failed result
            result = PipelineResult(
                pipeline_id=pipeline_id,
                success=False,
                processor_results=[],
                final_data=None,
                start_time=start_time,
                end_time=datetime.now(),
                metadata={'error': str(e)}
            )
            
            self._execution_history.append(result)
            raise
    
    def interrupt_pipeline_at(self, save_point_id: str) -> None:
        """
        Request pipeline interruption at specified save point.
        
        Args:
            save_point_id: Save point identifier where to interrupt
        """
        self.logger.info(f"Requesting pipeline interrupt at save point: {save_point_id}")
        
        if self._status == PipelineStatus.RUNNING:
            self._status = PipelineStatus.PAUSED
            # In real implementation, would signal the running pipeline
            # For now, just log the request
            self.logger.info("Pipeline interrupt requested")
        else:
            self.logger.warning(f"Cannot interrupt pipeline in status: {self._status}")
    
    def resume_pipeline_from_checkpoint(self, save_point_id: Optional[str] = None) -> None:
        """
        Resume pipeline execution from save point.
        
        Args:
            save_point_id: Optional specific save point to resume from
        """
        self.logger.info(f"Resuming pipeline from save point: {save_point_id or 'latest'}")
        
        if self._status == PipelineStatus.PAUSED:
            self._status = PipelineStatus.RUNNING
            # In real implementation, would resume from save point
            self.logger.info("Pipeline resumed")
        else:
            self.logger.warning(f"Cannot resume pipeline in status: {self._status}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary with comprehensive pipeline status
        """
        status = {
            "status": self._status.value,
            "pipeline_id": self._current_config.pipeline_id if self._current_config else None,
            "has_current_pipeline": self._current_pipeline is not None,
            "execution_history_count": len(self._execution_history),
            "last_execution": None
        }
        
        if self._execution_history:
            last_result = self._execution_history[-1]
            status["last_execution"] = {
                "pipeline_id": last_result.pipeline_id,
                "success": last_result.success,
                "duration_seconds": last_result.duration_seconds,
                "end_time": last_result.end_time.isoformat()
            }
        
        return status
    
    def get_processor_capabilities_summary(self, processors: List[BaseProcessor]) -> Dict[str, Any]:
        """
        Get summary of processor capabilities in the pipeline.
        
        Args:
            processors: List of processors to analyze
            
        Returns:
            Dictionary with capability analysis
        """
        summary = {
            "total_processors": len(processors),
            "capability_distribution": {},
            "processor_details": [],
            "pipeline_capabilities": set()
        }
        
        capability_counts = {}
        
        for processor in processors:
            capabilities = processor.get_capabilities()
            processor_type = processor.get_processor_type().value
            
            # Track processor details
            summary["processor_details"].append({
                "type": processor_type,
                "capabilities": capabilities
            })
            
            # Count capabilities
            for capability in capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
                summary["pipeline_capabilities"].add(capability)
        
        summary["capability_distribution"] = capability_counts
        summary["pipeline_capabilities"] = list(summary["pipeline_capabilities"])
        
        return summary
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
        self.logger.info("Cleared pipeline execution history")
    
    def get_execution_history(self) -> List[PipelineResult]:
        """Get the execution history."""
        return self._execution_history.copy()
    
    def _create_processor_node(self, processor: BaseProcessor, subgraph: CompiledStateGraph):
        """Create a pipeline node that wraps a processor subgraph."""
        
        def processor_node(state: ProcessorState) -> ProcessorState:
            """Execute processor subgraph and return updated state."""
            try:
                # Execute the processor's subgraph
                result_state = subgraph.invoke(state)
                
                # Log execution
                processor_type = processor.get_processor_type().value
                success = result_state.get('status') == 'success'
                self.logger.info(f"Processor {processor_type} completed: success={success}")
                
                return result_state
                
            except Exception as e:
                self.logger.error(f"Processor {processor.get_processor_type().value} failed: {e}")
                state['error'] = e
                state['status'] = 'error'
                return state
        
        return processor_node
    
    def _connect_processors_sequentially(self, graph: StateGraph, processors: List[BaseProcessor]) -> None:
        """Connect processors in sequential order."""
        processor_names = [f"{proc.get_processor_type().value}_{i}" for i, proc in enumerate(processors)]
        
        # Connect START to first processor
        if processor_names:
            graph.add_edge(START, processor_names[0])
        
        # Connect processors sequentially
        for i in range(len(processor_names) - 1):
            graph.add_edge(processor_names[i], processor_names[i + 1])
        
        # Connect last processor to END
        if processor_names:
            graph.add_edge(processor_names[-1], END)
    
    def _add_checkpointing_support(self, graph: StateGraph, config: PipelineConfig) -> None:
        """Add checkpointing support to the pipeline."""
        # In real implementation, would configure LangGraph checkpointing
        # For now, just log that checkpointing is enabled
        self.logger.info(f"Checkpointing enabled for pipeline: {config.pipeline_id}")
    
    def validate_pipeline_configuration(self, processors: List[BaseProcessor]) -> Dict[str, Any]:
        """
        Validate pipeline configuration and processor compatibility.
        
        Args:
            processors: List of processors to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "processor_analysis": {}
        }
        
        if not processors:
            validation["valid"] = False
            validation["issues"].append("No processors provided")
            return validation
        
        # Check processor types
        processor_types = [proc.get_processor_type() for proc in processors]
        type_counts = {}
        for proc_type in processor_types:
            type_counts[proc_type] = type_counts.get(proc_type, 0) + 1
        
        # Warn about duplicate processor types
        for proc_type, count in type_counts.items():
            if count > 1:
                validation["warnings"].append(f"Multiple instances of {proc_type.value} processor")
        
        # Validate individual processors
        for i, processor in enumerate(processors):
            proc_analysis = {
                "capabilities": processor.get_capabilities(),
                "type": processor.get_processor_type().value,
                "valid": True,
                "issues": []
            }
            
            # Check if processor can be created successfully
            try:
                self._processor_factory.validate_processor_capabilities(processor)
            except Exception as e:
                proc_analysis["valid"] = False
                proc_analysis["issues"].append(f"Capability validation failed: {e}")
                validation["valid"] = False
                validation["issues"].append(f"Processor {i} ({processor.get_processor_type().value}) validation failed")
            
            validation["processor_analysis"][f"processor_{i}"] = proc_analysis
        
        return validation 