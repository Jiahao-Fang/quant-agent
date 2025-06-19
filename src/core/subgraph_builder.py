"""
Dynamic LangGraph subgraph builder based on processor capabilities.
"""

from typing import TYPE_CHECKING, Any, Dict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph

from .base_processor import ProcessorState
from .workflow_nodes import ProcessNode, DebugNode, EvalNode, InterruptNode

if TYPE_CHECKING:
    from .base_processor import BaseProcessor


class SubgraphBuilder:
    """
    Dynamically generates LangGraph subgraphs based on processor capabilities.
    
    Analyzes processor decorators and builds appropriate workflow with:
    - Core processing node (always present)
    - Debug nodes (if @debuggable)
    - Evaluation nodes (if @evaluable)  
    - Interrupt nodes (if @interruptible) - inserted before every major step
    - Monitoring integration (if @observable)
    """
    
    def create_subgraph(self, processor: "BaseProcessor") -> CompiledGraph:
        """
        Create LangGraph subgraph based on processor capabilities.
        
        Args:
            processor: The processor to create subgraph for
            
        Returns:
            Complete CompiledGraph that can be used as a node in larger workflows
        """
        # Create base graph
        graph = StateGraph(ProcessorState)
        
        # Get processor capabilities
        capabilities = processor.get_capabilities()
        
        # Add core processing node (always present)
        process_node = ProcessNode(processor)
        graph.add_node("process_core", process_node.execute)
        
        # Build workflow based on capabilities
        if not capabilities:
            # Simple workflow: START -> process_core -> END
            self._build_simple_workflow(graph)
        else:
            # Complex workflow with capabilities
            self._build_capability_workflow(graph, processor, capabilities)
        
        return graph.compile()
    
    def _build_simple_workflow(self, graph: StateGraph) -> None:
        """Build simple workflow without any capabilities."""
        graph.add_edge(START, "process_core")
        graph.add_edge("process_core", END)
    
    def _build_capability_workflow(
        self, 
        graph: StateGraph, 
        processor: "BaseProcessor",
        capabilities: List[str]
    ) -> None:
        """Build complex workflow with capabilities."""
        
        # Add capability nodes
        if "interruptible" in capabilities:
            self._add_interrupt_capability(graph, processor)
        
        if "debuggable" in capabilities:
            self._add_debug_capability(graph, processor)
        
        if "evaluable" in capabilities:
            self._add_eval_capability(graph, processor)
        
        if "observable" in capabilities:
            self._add_monitoring_capability(graph, processor)
        
        # Build the workflow logic with interrupt checks at every step
        self._connect_workflow_nodes_with_interrupts(graph, capabilities)
    
    def _add_interrupt_capability(
        self, 
        graph: StateGraph, 
        processor: "BaseProcessor"
    ) -> None:
        """Add interrupt checking nodes for each major step."""
        interrupt_node = InterruptNode(processor)
        
        # Always add handle_interrupt
        graph.add_node("handle_interrupt", interrupt_node.handle_interrupt)
        
        # Add interrupt check nodes based on other capabilities
        capabilities = processor.get_capabilities()
        
        # Always add start interrupt check
        graph.add_node("check_interrupt_start", interrupt_node.check_interrupt)
        
        # Add debug interrupt check if debuggable
        if "debuggable" in capabilities:
            graph.add_node("check_interrupt_debug", interrupt_node.check_interrupt)
        
        # Add eval interrupt check if evaluable
        if "evaluable" in capabilities:
            graph.add_node("check_interrupt_eval", interrupt_node.check_interrupt)
    
    def _add_debug_capability(
        self, 
        graph: StateGraph, 
        processor: "BaseProcessor"
    ) -> None:
        """Add debug handling nodes."""
        debug_node = DebugNode(processor)
        graph.add_node("debug_error", debug_node.execute)
        graph.add_node("check_debug_retry", debug_node.check_retry)
    
    def _add_eval_capability(
        self, 
        graph: StateGraph, 
        processor: "BaseProcessor"
    ) -> None:
        """Add evaluation nodes."""
        eval_node = EvalNode(processor)
        graph.add_node("evaluate_result", eval_node.execute)
        graph.add_node("check_eval_retry", eval_node.check_retry)
    
    def _add_monitoring_capability(
        self, 
        graph: StateGraph, 
        processor: "BaseProcessor"
    ) -> None:
        """Add monitoring integration to existing nodes."""
        # Monitoring is handled by wrapping existing nodes
        # This is implemented in the node classes themselves
        pass
    
    def _connect_workflow_nodes_with_interrupts(
        self, 
        graph: StateGraph, 
        capabilities: List[str]
    ) -> None:
        """Connect workflow nodes with interrupt checks before every major step if interruptible."""
        
        is_interruptible = "interruptible" in capabilities
        
        # Start point
        if is_interruptible:
            graph.add_edge(START, "check_interrupt_start")
            graph.add_conditional_edges(
                "check_interrupt_start",
                self._should_interrupt,
                {
                    "interrupt": "handle_interrupt",
                    "continue": "process_core"
                }
            )
            graph.add_edge("handle_interrupt", END)
        else:
            graph.add_edge(START, "process_core")
        
        # Connect processing to next step
        if "debuggable" in capabilities:
            graph.add_conditional_edges(
                "process_core",
                self._has_error,
                {
                    "error": "check_interrupt_debug" if is_interruptible else "debug_error",
                    "success": self._get_success_target(capabilities, is_interruptible)
                }
            )
            
            # Debug flow
            if is_interruptible:
                # With interrupt check before debug
                graph.add_conditional_edges(
                    "check_interrupt_debug",
                    self._should_interrupt,
                    {
                        "interrupt": "handle_interrupt",
                        "continue": "debug_error"
                    }
                )
                
                graph.add_conditional_edges(
                    "debug_error",
                    self._should_retry_after_debug,
                    {
                        "retry": "check_interrupt_start",  # Back to start with interrupt check
                        "fail": END
                    }
                )
            else:
                # Direct debug flow without interrupt checks
                graph.add_conditional_edges(
                    "debug_error",
                    self._should_retry_after_debug,
                    {
                        "retry": "process_core",  # Direct retry
                        "fail": END
                    }
                )
        else:
            # No debug capability, go directly to next step
            graph.add_edge("process_core", self._get_success_target(capabilities, is_interruptible))
        
        # Connect evaluation flow
        if "evaluable" in capabilities:
            if is_interruptible:
                # With interrupt check before evaluation
                graph.add_conditional_edges(
                    "check_interrupt_eval",
                    self._should_interrupt,
                    {
                        "interrupt": "handle_interrupt",
                        "continue": "evaluate_result"
                    }
                )
                
                graph.add_conditional_edges(
                    "evaluate_result",
                    self._eval_passed,
                    {
                        "pass": END,
                        "fail": "check_eval_retry"
                    }
                )
                
                graph.add_conditional_edges(
                    "check_eval_retry",
                    self._should_retry_after_eval,
                    {
                        "retry": "check_interrupt_start",  # Back to start with interrupt check
                        "fail": END
                    }
                )
            else:
                # Direct evaluation flow without interrupt checks
                graph.add_conditional_edges(
                    "evaluate_result",
                    self._eval_passed,
                    {
                        "pass": END,
                        "fail": "check_eval_retry"
                    }
                )
                
                graph.add_conditional_edges(
                    "check_eval_retry",
                    self._should_retry_after_eval,
                    {
                        "retry": "process_core",  # Direct retry
                        "fail": END
                    }
                )
    
    def _get_success_target(self, capabilities: List[str], is_interruptible: bool) -> str:
        """Get the target node for successful processing."""
        if "evaluable" in capabilities:
            return "check_interrupt_eval" if is_interruptible else "evaluate_result"
        return END
    
    # ================================
    # CONDITIONAL EDGE FUNCTIONS
    # ================================
    
    def _should_interrupt(self, state: ProcessorState) -> str:
        """Check if processing should be interrupted."""
        return "interrupt" if state.get("interrupt_requested", False) else "continue"
    
    def _has_error(self, state: ProcessorState) -> str:
        """Check if processing resulted in an error."""
        return "error" if state.get("error") is not None else "success"
    
    def _should_retry_after_debug(self, state: ProcessorState) -> str:
        """Check if processing should be retried after debug."""
        if not state.get("should_retry", False):
            return "fail"
        
        # Check retry limits
        debug_config = state.get("debug_config", {})
        max_retries = debug_config.get("max_retries", 3)
        debug_retry_count = state.get("debug_retry_count", 0)
        
        return "retry" if debug_retry_count < max_retries else "fail"
    
    def _eval_passed(self, state: ProcessorState) -> str:
        """Check if evaluation passed."""
        return "pass" if state.get("eval_passed", False) else "fail"
    
    def _should_retry_after_eval(self, state: ProcessorState) -> str:
        """Check if processing should be retried after evaluation."""
        # Check retry limits
        eval_config = state.get("eval_config", {})
        max_retries = eval_config.get("max_retries", 3)
        eval_retry_count = state.get("eval_retry_count", 0)
        
        return "retry" if eval_retry_count < max_retries else "fail" 