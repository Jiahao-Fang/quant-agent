"""
Main entry point for the quant factor pipeline.
Coordinates the overall workflow of factor research and development.
"""

from typing import Dict, TypedDict, Optional
import pykx as kx
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import templates from prompt_lib
from prompt_lib.factor_build_lead import FACTOR_BUILD_LEAD_TEMPLATE
from prompt_lib.data_fields_description import DATA_FIELDS_DESCRIPTION


# Import subgraphs
from data_fetch import create_data_fetch_graph
from feature_build import create_feature_build_graph

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# State type for the full pipeline
class EntryGraphState(TypedDict):
    human_input: str  # Original factor description from user
    feature_description: str  # Processed feature description
    query: str  # Generated query for data fetching
    data: Optional[Dict[str, kx.Table]]  # Fetched data tables
    error: Optional[str]  # Error message if any
    result: Optional[kx.Table]  # Final feature table

def demand_analysis(state: EntryGraphState) -> Dict:
    """Analyze user's demand and generate detailed feature description"""
    prompt = FACTOR_BUILD_LEAD_TEMPLATE.format(
        factor_description=state["human_input"], 
        data_fields_description=DATA_FIELDS_DESCRIPTION
    )
    feature_description = llm.invoke(prompt)
    return {"feature_description": feature_description.content}

def should_continue(state: EntryGraphState) -> str:
    """Determine if the pipeline should continue or stop"""
    if state.get("error"):
        return "end"
    return "feature_build"

def create_entry_graph() -> StateGraph:
    """
    Create and return the main entry graph that coordinates the entire pipeline.
    
    Returns:
        StateGraph: Compiled entry graph
    """
    # Create graph builder
    entry_builder = StateGraph(EntryGraphState)
    
    # Create subgraphs
    data_fetch_graph = create_data_fetch_graph()
    feature_build_graph = create_feature_build_graph()
    
    # Add nodes
    entry_builder.add_node("demand_analysis", demand_analysis)
    entry_builder.add_node("data_fetch", data_fetch_graph)
    entry_builder.add_node("feature_build", feature_build_graph)
    
    # Add edges
    entry_builder.add_edge(START, "demand_analysis")
    entry_builder.add_edge("demand_analysis", "data_fetch")
    
    # Add conditional edges
    entry_builder.add_conditional_edges(
        "data_fetch",
        should_continue,
        {
            "feature_build": "feature_build",
            "end": END
        }
    )
    entry_builder.add_edge("feature_build", END)
    
    # Compile and return
    return entry_builder.compile()

def run_pipeline(human_input: str) -> Dict:
    """
    Run the complete factor research pipeline.
    
    Args:
        human_input: Natural language description of the desired factor
        
    Returns:
        Dict containing the final results and status
    """
    initial_state = {
        "human_input": human_input,
        "feature_description": "",
        "query": "",
        "data": None,
        "error": None,
        "result": None
    }
    
    # Get the graph
    graph = create_entry_graph()
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Return results
    if final_state.get("error"):
        return {
            "success": False,
            "error": final_state["error"],
            "result": None
        }
    else:
        return {
            "success": True,
            "result": final_state["result"],
            "feature_description": final_state["feature_description"]
        }
