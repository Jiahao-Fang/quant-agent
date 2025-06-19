"""
Enhanced Feature Build module with debug logic for the quant factor pipeline.
Handles feature code generation, validation, debugging and execution.
"""

from typing import TypedDict, Dict, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import traceback
import pykx as kx
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import templates from prompt_lib
from prompt_lib.factor_build_dev import FACTOR_BUILD_DEV_TEMPLATE
from prompt_lib.data_build_debug_1 import DATA_BUILD_DEBUG_1_TEMPLATE
from prompt_lib.data_build_debug_2 import DATA_BUILD_DEBUG_2_TEMPLATE

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Enhanced Feature build state definition
class FeatureBuildState(TypedDict):
    query: str 
    feature_description: str
    feature_code: str
    data: Dict[str, kx.Table]
    feature_table: Optional[kx.Table]
    
    # Debug related fields
    debug_code: Optional[str]
    debug_info: Optional[Dict]  # 新的统一格式的 debug 信息
    error_message: Optional[str]
    execution_result: Optional[str]
    
    # Retry management
    current_retry: int
    max_retries: int
    eval_stage: int  # 0: initial, 1: gen_debug, 2: exec_debug, 3: fix
    is_successful: bool
    debug_history: List[str]
    code_validated: bool
    code_history: List[Dict]  # 新增：追踪代码演化历史

# Prompt templates will be defined elsewhere
eval_prompt_stage1 = ""
eval_prompt_stage2 = ""

# Helper function to extract code from response
def extract_code_from_response(response_content: str) -> str:
    """Extract Python code from LLM response"""
    # Remove markdown code blocks if present
    if "```python" in response_content:
        start = response_content.find("```python") + 9
        end = response_content.find("```", start)
        return response_content[start:end].strip()
    elif "```" in response_content:
        start = response_content.find("```") + 3
        end = response_content.find("```", start)
        return response_content[start:end].strip()
    return response_content.strip()

# Node functions
def generate_feature_code(state: FeatureBuildState) -> Dict:
    """Generate feature calculation code based on description and query"""
    prompt = FACTOR_BUILD_DEV_TEMPLATE.format(
        factor_description=state['feature_description'],
        data_description=state['query']
    )
    response = llm.invoke(prompt)
    print("Generated code:", response.content)
    
    code = extract_code_from_response(response.content)
    code = f"{code}\nresult = compute_factor(data_dict)"
    
    return {
        "feature_code": code,
        "current_retry": 0,
        "max_retries": 3,
        "eval_stage": 0,
        "is_successful": False,
        "debug_history": [],
        "code_validated": False
    }

def eval_code(state: FeatureBuildState) -> Dict:
    """Evaluate code by generating debug version and executing it"""
    print(f"Eval: Validating code (Retry {state.get('current_retry', 0)})")
    
    # 初始化标准的 debug_info 结构
    debug_info = {
        "steps": [],
        "variables": {},
        "errors": [],
        "success": False,
        "final_result": None
    }
    
    # 获取当前代码历史
    code_history = state.get('code_history', [])
    
    try:
        # Step 1: Generate debug version of the code
        prompt = DATA_BUILD_DEBUG_1_TEMPLATE.format(original_code=state['feature_code'])
        response = llm.invoke(prompt)
        debug_code = extract_code_from_response(response.content)
        debug_info["steps"].append("Generated debug code")
        
        # Step 2: Execute debugging version
        exec_globals = {
            'kx': kx,
            'np': np, 
            'pd': pd,
            'traceback': traceback
        }
        local_ns = {}
        
        exec(debug_code, exec_globals, local_ns)
        debug_info["steps"].append("Executed debug code")
        
        # Get debug function and execute it
        debug_func = local_ns.get('compute_factor_debug')
        if debug_func:
            try:
                result = debug_func(state['data'])
                debug_info["variables"] = result.get("variables", {})
                debug_info["final_result"] = result.get("result")
                debug_info["steps"].extend(result.get("steps", []))
                
                if not result.get("errors"):
                    debug_info["success"] = True
                    print("✅ Code validation successful!")
                    
                    # 添加成功的代码到历史
                    code_history.append({
                        "code": state['feature_code'],
                        "status": "success",
                        "retry": state.get('current_retry', 0),
                        "debug_info": debug_info
                    })
                else:
                    debug_info["errors"].extend(result.get("errors", []))
                    print("❌ Code validation failed, needs debugging")
                    
                    # 添加失败的代码到历史
                    code_history.append({
                        "code": state['feature_code'],
                        "status": "failed",
                        "retry": state.get('current_retry', 0),
                        "debug_info": debug_info
                    })
                
            except Exception as e:
                debug_info["errors"].append(f"Debug function execution error: {str(e)}")
                debug_info["steps"].append("Debug function execution failed")
                
                # 添加错误的代码到历史
                code_history.append({
                    "code": state['feature_code'],
                    "status": "failed",
                    "retry": state.get('current_retry', 0),
                    "debug_info": debug_info
                })
        else:
            debug_info["errors"].append("Debug function 'compute_factor_debug' not found")
            debug_info["steps"].append("Failed to locate debug function")
            
            # 添加错误的代码到历史
            code_history.append({
                "code": state['feature_code'],
                "status": "failed",
                "retry": state.get('current_retry', 0),
                "debug_info": debug_info
            })
        
        return {
            "debug_code": debug_code,
            "debug_info": debug_info,
            "code_validated": debug_info["success"],
            "is_successful": debug_info["success"],
            "code_history": code_history
        }
        
    except Exception as e:
        error_detail = traceback.format_exc()
        debug_info["errors"].append(f"Evaluation error: {str(e)}")
        debug_info["steps"].append("Evaluation failed with exception")
        
        # 添加异常错误的代码到历史
        code_history.append({
            "code": state['feature_code'],
            "status": "failed",
            "retry": state.get('current_retry', 0),
            "debug_info": debug_info
        })
        
        return {
            "debug_code": state.get('debug_code'),
            "debug_info": debug_info,
            "code_validated": False,
            "is_successful": False,
            "code_history": code_history
        }

def debug_fix_code(state: FeatureBuildState) -> Dict:
    """Debug: Analyze validation results and generate fixed code"""
    print("Debug Stage: Analyzing validation results and generating fix")
    
    # Prepare debug information for analysis
    if state.get('debug_info'):
        debug_info = state['debug_info']
        error_messages = debug_info.get('errors', [])
        debug_variables = debug_info.get('variables', {})
        completed_steps = debug_info.get('steps', [])
    else:
        error_messages = [state.get('error_message', 'Unknown error')]
        debug_variables = {}
        completed_steps = []
    
    prompt = DATA_BUILD_DEBUG_2_TEMPLATE.format(
        original_code=state['feature_code'],
        error_messages=str(error_messages),
        debug_variables=str(debug_variables),
        completed_steps=str(completed_steps),
        factor_description=state['feature_description'],
        data_description=state['query'],
        current_retry=state.get('current_retry', 0),
        max_retries=state.get('max_retries', 3)
    )
    
    response = llm.invoke(prompt)
    fixed_code = extract_code_from_response(response.content)
    
    # Update debug history
    debug_history = state.get("debug_history", [])
    debug_history.append(f"Retry {state.get('current_retry', 0)}: {str(error_messages)}")
    
    return {
        "feature_code": fixed_code,
        "debug_history": debug_history,
        "current_retry": state.get("current_retry", 0) + 1,
        "eval_stage": 0,  # Reset to start eval cycle again
        "debug_info": None,
        "code_validated": False
    }

def execute_final_code(state: FeatureBuildState) -> Dict:
    """Execute the validated code (clean version without debug output)"""
    print("Executing final validated code...")
    
    try:
        # Create execution environment
        exec_globals = {
            'kx': kx,
            'np': np,
            'pd': pd,
            'data_dict': state['data']
        }
        local_ns = {}
        print(state["feature_code"])
        # Execute the clean code string (original feature_code, not debug version)
        exec(state["feature_code"], exec_globals, local_ns)
        
        # Get the result
        feature_table = local_ns.get("result")
        
        if feature_table is not None:
            print(f"✅ Final execution successful. Result type: {type(feature_table)}")
            return {
                "feature_table": feature_table,
                "execution_result": f"Success: Generated feature table with type {type(feature_table)}",
                "error_message": None,
                "is_successful": True
            }
        else:
            print("❌ Final execution failed: 'result' variable not found")
            return {
                "feature_table": None,
                "execution_result": None,
                "error_message": "Final execution completed but 'result' variable not found",
                "is_successful": False
            }
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"❌ Final execution error: {str(e)}")
        
        return {
            "feature_table": None,
            "execution_result": None,
            "error_message": f"Final execution error: {str(e)}\n\nDetails:\n{error_detail}",
            "is_successful": False
        }

# Routing functions
def should_continue_after_generate(state: FeatureBuildState) -> str:
    """Route after code generation - always go to eval first"""
    return "eval_code"

def should_continue_after_eval(state: FeatureBuildState) -> str:
    """Route after eval - check if validation passed"""
    if state.get("code_validated", False):
        return "execute_final"  # Code is validated, execute clean version
    else:
        # Check retry limit
        if state.get("current_retry", 0) >= state.get("max_retries", 3):
            print(f"Max retries ({state.get('max_retries', 3)}) reached. Ending with failure.")
            return "end"
        else:
            return "debug_fix"  # Need to fix code

def should_continue_after_debug(state: FeatureBuildState) -> str:
    """Route after debug fix - go back to eval cycle"""
    return "eval_code"

def should_continue_after_final_execute(state: FeatureBuildState) -> str:
    """Route after final execution - always end"""
    return "end"

def create_feature_build_graph() -> StateGraph:
    """
    Create and return the feature building graph with debug capabilities.
    
    Returns:
        StateGraph: Compiled feature building graph
    """
    # Create graph builder
    fb_builder = StateGraph(FeatureBuildState)
    
    # Add all nodes
    fb_builder.add_node("generate_code", generate_feature_code)
    fb_builder.add_node("eval_code", eval_code)
    fb_builder.add_node("debug_fix", debug_fix_code)
    fb_builder.add_node("execute_final", execute_final_code)
    
    # Add edges following the flow:
    # generate -> eval_code -> (if validated) execute_final
    #                      -> (if not) debug_fix -> eval_code (cycle)
    fb_builder.add_edge(START, "generate_code")
    fb_builder.add_edge("generate_code", "eval_code")
    
    fb_builder.add_conditional_edges(
        "eval_code",
        should_continue_after_eval,
        {
            "execute_final": "execute_final",
            "debug_fix": "debug_fix",
            "end": END
        }
    )
    fb_builder.add_edge("debug_fix", "eval_code")
    fb_builder.add_edge("execute_final", END)
    
    # Compile and return
    return fb_builder.compile()

def run_feature_build(query: str, feature_description: str, data: Dict[str, kx.Table]) -> Dict:
    """
    Run the enhanced feature build process with debug capabilities.
    
    Args:
        query: Query string used to fetch the data
        feature_description: Natural language description of the feature
        data: Dictionary of KDB tables
        
    Returns:
        Dict containing build results and status
    """
    initial_state = {
        "query": query,
        "feature_description": feature_description,
        "feature_code": "",
        "data": data,
        "feature_table": None,
        
        # Debug fields
        "debug_code": None,
        "debug_info": None,
        "error_message": None,
        "execution_result": None,
        
        # Retry management
        "current_retry": 0,
        "max_retries": 3,
        "eval_stage": 0,
        "is_successful": False,
        "debug_history": [],
        "code_validated": False,
        "code_history": []  # 新增：代码历史追踪
    }
    
    # Get the graph
    graph = create_feature_build_graph()
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Return results
    if final_state.get("is_successful", False):
        print("✅ Feature build completed successfully!")
        return {
            "success": True,
            "feature_table": final_state["feature_table"],
            "final_code": final_state["feature_code"],
            "retries_used": final_state.get("current_retry", 0)
        }
    else:
        print("❌ Feature build failed after all retries")
        return {
            "success": False,
            "feature_table": None,
            "final_code": final_state["feature_code"],
            "error_message": final_state.get("error_message", "Unknown error"),
            "debug_history": final_state.get("debug_history", []),
            "retries_used": final_state.get("current_retry", 0)
        }

# Display the graph structure
print("Enhanced Feature Build Graph with Debug Logic created successfully!")
print("Flow: generate_code -> eval_code -> [validated?] -> execute_final")
print("      [not validated] -> debug_fix -> eval_code (cycle until validated or max retries)")