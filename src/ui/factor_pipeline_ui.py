"""
Streamlit UI for Quant Factor Pipeline
Displays the complete workflow with interactive components
"""

import streamlit as st
import pandas as pd
import pykx as kx
import numpy as np
import json
from typing import Dict, List, Optional
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entry import run_pipeline
from feature_build import FeatureBuildState

# Configure page
st.set_page_config(
    page_title="Quant Factor Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        color: #2e7d32;
        font-weight: bold;
    }
    .status-error {
        color: #d32f2f;
        font-weight: bold;
    }
    .status-running {
        color: #f57c00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'pipeline_state' not in st.session_state:
        st.session_state.pipeline_state = {}
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_paused' not in st.session_state:
        st.session_state.pipeline_paused = False
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'code_history' not in st.session_state:
        st.session_state.code_history = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = ""
    if 'step_status' not in st.session_state:
        st.session_state.step_status = {}
    if 'query_info' not in st.session_state:
        st.session_state.query_info = ""
    if 'step_results_history' not in st.session_state:
        st.session_state.step_results_history = []
    if 'interrupt_requested' not in st.session_state:
        st.session_state.interrupt_requested = False
    if 'current_checkpoint' not in st.session_state:
        st.session_state.current_checkpoint = None
    if 'user_intervention' not in st.session_state:
        st.session_state.user_intervention = ""
    if 'intervention_type' not in st.session_state:
        st.session_state.intervention_type = ""

class PipelineInterrupt(Exception):
    """Custom exception for pipeline interruption"""
    def __init__(self, message, checkpoint_data=None):
        self.message = message
        self.checkpoint_data = checkpoint_data
        super().__init__(self.message)

def add_step_result(step_name: str, result_data: dict, status: str = "completed"):
    """Add a step result to the history"""
    step_result = {
        "step_name": step_name,
        "timestamp": time.time(),
        "status": status,
        "data": result_data
    }
    st.session_state.step_results_history.append(step_result)

def display_step_results_history():
    """Display all step results in chronological order"""
    st.markdown('<div class="section-header">📜 Execution History</div>', unsafe_allow_html=True)
    
    if not st.session_state.step_results_history:
        st.info("No execution history yet. Run the pipeline to see step-by-step results.")
        return
    
    for i, step_result in enumerate(st.session_state.step_results_history):
        step_name = step_result["step_name"]
        status = step_result["status"]
        data = step_result["data"]
        timestamp = step_result["timestamp"]
        
        # Create an expander for each step
        status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "🔄"
        
        with st.expander(f"{status_icon} {step_name} ({time.strftime('%H:%M:%S', time.localtime(timestamp))})"):
            if step_name == "Demand Analysis":
                st.write("**Feature Description:**")
                st.write(data.get("feature_description", ""))
                
            elif step_name == "Query Generation":
                st.write("**Generated Query:**")
                st.code(data.get("query", ""), language='json')
                
            elif step_name == "Data Fetching":
                st.write("**Fetched Data:**")
                data_dict = data.get("data", {})
                if data_dict:
                    for key, table in data_dict.items():
                        st.write(f"- **{key}**: {table.pd().shape if hasattr(table, 'pd') else 'Data available'}")
                
            elif step_name == "Code Generation":
                st.write("**Generated Code:**")
                st.code(data.get("code", ""), language='python')
                
            elif step_name == "Code Evaluation":
                st.write("**Evaluation Result:**")
                debug_info = data.get("debug_info", {})
                if debug_info.get("errors"):
                    st.error("Errors found:")
                    for error in debug_info["errors"]:
                        st.write(f"- {error}")
                else:
                    st.success("Code validation successful!")
                
            elif step_name == "Code Fix":
                st.write("**Fixed Code:**")
                st.code(data.get("fixed_code", ""), language='python')
                
            elif step_name == "Final Execution":
                st.write("**Execution Result:**")
                if data.get("feature_table") is not None:
                    st.success("Feature table generated successfully!")
                    df = data["feature_table"].pd() if hasattr(data["feature_table"], 'pd') else data["feature_table"]
                    st.write(f"Shape: {df.shape}")
                    st.dataframe(df.head(), use_container_width=True)
                else:
                    st.error("Execution failed")

def display_pipeline_controls():
    """Display pipeline control buttons in sidebar"""
    st.sidebar.markdown("### 🎮 Pipeline Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.session_state.pipeline_running and not st.session_state.pipeline_paused:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.interrupt_requested = True
                st.info("Pause requested. Will stop after current step completes.")
        
        elif st.session_state.pipeline_paused:
            if st.button("▶️ Resume", use_container_width=True):
                st.session_state.pipeline_paused = False
                st.session_state.interrupt_requested = False
                st.rerun()
    
    with col2:
        if st.button("🛑 Stop", use_container_width=True):
            st.session_state.pipeline_running = False
            st.session_state.pipeline_paused = False
            st.session_state.interrupt_requested = False
            st.session_state.current_checkpoint = None
            st.warning("Pipeline stopped.")

def display_intervention_panel():
    """Display user intervention panel when paused"""
    if st.session_state.pipeline_paused and st.session_state.current_checkpoint:
        st.markdown("### 🛠️ User Intervention")
        
        intervention_type = st.selectbox(
            "Select intervention type:",
            ["", "Modify Feature Code", "Change Data Query", "Adjust Parameters", "Custom Instruction"],
            key="intervention_type_select"
        )
        
        if intervention_type:
            st.session_state.intervention_type = intervention_type
            
            user_input = st.text_area(
                f"Enter your {intervention_type.lower()}:",
                height=150,
                placeholder="Describe what you want to change or improve...",
                key="user_intervention_input"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Apply Changes", type="primary"):
                    if user_input.strip():
                        st.session_state.user_intervention = user_input.strip()
                        st.session_state.pipeline_paused = False
                        st.success("Changes applied! Resuming pipeline...")
                        st.rerun()
                    else:
                        st.error("Please enter your intervention instructions.")
            
            with col2:
                if st.button("🔄 Resume Without Changes"):
                    st.session_state.intervention_type = ""
                    st.session_state.user_intervention = ""
                    st.session_state.pipeline_paused = False
                    st.info("Resuming pipeline without changes...")
                    st.rerun()

def check_interrupt_point(step_name: str, checkpoint_data: dict):
    """Check if pipeline should be interrupted at this point"""
    if st.session_state.interrupt_requested:
        st.session_state.pipeline_paused = True
        st.session_state.interrupt_requested = False
        st.session_state.current_checkpoint = {
            "step_name": step_name,
            "data": checkpoint_data,
            "timestamp": time.time()
        }
        raise PipelineInterrupt(f"Pipeline paused at {step_name}", checkpoint_data)

def display_kdb_data_section(data_dict: Dict[str, kx.Table]):
    """Display KDB data tables with interactive key selection"""
    st.markdown('<div class="section-header">📊 Data Fetching Results</div>', unsafe_allow_html=True)
    
    if not data_dict:
        st.info("No data available yet. Run the pipeline to see results.")
        return
    
    # Create tabs for different data keys
    table_keys = list(data_dict.keys())
    
    if len(table_keys) == 1:
        # Single table - no tabs needed
        key = table_keys[0]
        st.write(f"**Table: {key}**")
        try:
            # Convert kx.Table to pandas for display
            df = data_dict[key].pd()
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying table {key}: {str(e)}")
    else:
        # Multiple tables - use tabs
        tabs = st.tabs([f"📋 {key}" for key in table_keys])
        
        for i, key in enumerate(table_keys):
            with tabs[i]:
                try:
                    # Convert kx.Table to pandas for display
                    df = data_dict[key].pd()
                    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show column info
                    with st.expander("Column Information"):
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Data Type': df.dtypes,
                            'Non-Null Count': df.count(),
                            'Null Count': df.isnull().sum()
                        })
                        st.dataframe(col_info, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error displaying table {key}: {str(e)}")

def display_code_evolution_section(code_history: List[Dict]):
    """Display feature code evolution with tabs for each iteration"""
    st.markdown('<div class="section-header">💻 Feature Code Evolution</div>', unsafe_allow_html=True)
    
    if not code_history:
        st.info("No code generation history yet. Run the pipeline to see code evolution.")
        return
    
    # Create tabs for different code versions
    tab_names = []
    for i, code_info in enumerate(code_history):
        status = code_info.get('status', 'unknown')
        retry_num = code_info.get('retry', 0)
        if i == 0:
            tab_names.append("🔧 Original")
        elif status == 'success':
            tab_names.append(f"✅ Retry {retry_num}")
        elif status == 'failed':
            tab_names.append(f"❌ Retry {retry_num}")
        else:
            tab_names.append(f"🔄 Retry {retry_num}")
    
    tabs = st.tabs(tab_names)
    
    for i, (tab, code_info) in enumerate(zip(tabs, code_history)):
        with tab:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.code(code_info.get('code', ''), language='python')
            
            with col2:
                st.write("**Status:**")
                status = code_info.get('status', 'unknown')
                if status == 'success':
                    st.markdown('<p class="status-success">✅ Success</p>', unsafe_allow_html=True)
                elif status == 'failed':
                    st.markdown('<p class="status-error">❌ Failed</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-running">🔄 Processing</p>', unsafe_allow_html=True)
                
                # Show debug info if available
                debug_info = code_info.get('debug_info')
                if debug_info:
                    st.write("**Debug Info:**")
                    with st.expander("View Details"):
                        if debug_info.get('errors'):
                            st.write("**Errors:**")
                            for error in debug_info['errors']:
                                st.error(error)
                        
                        if debug_info.get('steps'):
                            st.write("**Steps:**")
                            for step in debug_info['steps']:
                                st.write(f"• {step}")
                        
                        if debug_info.get('variables'):
                            st.write("**Variables:**")
                            st.json(debug_info['variables'])

def display_feature_table_section(feature_table: Optional[kx.Table]):
    """Display the final feature table"""
    st.markdown('<div class="section-header">📈 Final Feature Table</div>', unsafe_allow_html=True)
    
    if feature_table is None:
        st.info("No feature table generated yet. Run the pipeline to see results.")
        return
    
    try:
        # Convert to pandas for display
        df = feature_table.pd()
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Features", df.shape[1] - 1 if 'timestamp' in df.columns else df.shape[1])
        
        # Display data
        st.write("**Preview (First 20 rows):**")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Show column statistics
        with st.expander("📊 Column Statistics"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics.")
        
        # Show data types
        with st.expander("🔍 Data Types"):
            type_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(type_info, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying feature table: {str(e)}")

def display_pipeline_status(pipeline_state: Dict):
    """Display current pipeline status in sidebar"""
    st.sidebar.markdown("### 🔄 Pipeline Status")
    
    stages = [
        ("Demand Analysis", "feature_description"),
        ("Data Fetching", "data"),
        ("Feature Building", "feature_table")
    ]
    
    for stage_name, key in stages:
        if key in pipeline_state and pipeline_state[key] is not None:
            st.sidebar.markdown(f"✅ {stage_name}")
        elif st.session_state.pipeline_running:
            st.sidebar.markdown(f"🔄 {stage_name}")
        else:
            st.sidebar.markdown(f"⏳ {stage_name}")

def display_query_section():
    """Display the generated query information"""
    st.markdown('<div class="section-header">🔍 Generated Query</div>', unsafe_allow_html=True)
    
    if not st.session_state.query_info:
        st.info("No query generated yet. Run the pipeline to see the query.")
        return
    
    try:
        # Try to parse as JSON for better formatting
        import json
        query_data = json.loads(st.session_state.query_info)
        st.json(query_data)
    except:
        # If not JSON, display as text
        st.code(st.session_state.query_info, language='json')
    
    # Show query details if it's a JSON structure
    try:
        query_data = json.loads(st.session_state.query_info)
        if isinstance(query_data, list) and len(query_data) > 0:
            query_item = query_data[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Symbol", query_item.get('symbol', 'N/A'))
            with col2:
                st.metric("Exchange", query_item.get('exchange', 'N/A'))
            with col3:
                data_sources = query_item.get('data_sources', [])
                st.metric("Data Sources", len(data_sources))
    except:
        pass

def run_pipeline_with_realtime_updates(human_input: str):
    """Run the pipeline with real-time updates and interrupt support"""
    try:
        st.session_state.pipeline_running = True
        st.session_state.pipeline_paused = False
        st.session_state.code_history = []
        st.session_state.current_step = "Initializing..."
        st.session_state.step_status = {}
        
        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Check if resuming from checkpoint
        if st.session_state.current_checkpoint:
            return resume_from_checkpoint(status_placeholder, progress_bar)
        
        # Step 1: Demand Analysis
        try:
            with status_placeholder.container():
                st.info("🔄 Step 1: Analyzing your factor request...")
            progress_bar.progress(10)
            st.session_state.current_step = "Demand Analysis"
            
            # Import here to avoid circular imports
            from entry import demand_analysis
            
            # Initialize state
            initial_state = {
                "human_input": human_input,
                "feature_description": "",
                "query": "",
                "data": None,
                "error": None,
                "result": None
            }
            
            # Run demand analysis
            demand_result = demand_analysis(initial_state)
            feature_description = demand_result.get("feature_description", "")
            
            # Add to results history
            add_step_result("Demand Analysis", {
                "feature_description": feature_description,
                "human_input": human_input
            })
            
            st.session_state.pipeline_state['feature_description'] = feature_description
            st.session_state.step_status['demand_analysis'] = 'completed'
            
            with status_placeholder.container():
                st.success("✅ Step 1: Factor analysis completed!")
                with st.expander("View Feature Description"):
                    st.write(feature_description)
            
            progress_bar.progress(30)
            
            # Check for interrupt
            check_interrupt_point("Demand Analysis", {
                "feature_description": feature_description,
                "next_step": "Data Fetching"
            })
            
        except PipelineInterrupt as e:
            st.info(f"⏸️ Pipeline paused at: {e.message}")
            return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
        
        # Step 2: Data Fetching
        try:
            with status_placeholder.container():
                st.info("🔄 Step 2: Generating query and fetching data...")
            st.session_state.current_step = "Data Fetching"
            
            # Generate and run data fetch
            from data_fetch import create_data_fetch_graph
            data_fetch_graph = create_data_fetch_graph()
            
            data_state = {
                "feature_description": feature_description,
                "query": "",
                "data": {},
                "error": None
            }
            
            data_result = data_fetch_graph.invoke(data_state)
            
            # Add query generation to history
            if data_result.get("query"):
                add_step_result("Query Generation", {
                    "query": data_result["query"]
                })
                st.session_state.query_info = data_result["query"]
                st.session_state.step_status['query_generation'] = 'completed'
            
            # Add data fetching to history
            if data_result.get("data"):
                add_step_result("Data Fetching", {
                    "data": data_result["data"]
                })
                st.session_state.pipeline_state['data'] = data_result["data"]
                st.session_state.step_status['data_fetch'] = 'completed'
            
            with status_placeholder.container():
                if data_result.get("error"):
                    st.error(f"❌ Step 2: Data fetching failed: {data_result['error']}")
                    add_step_result("Data Fetching", {"error": data_result["error"]}, "failed")
                    return {"success": False, "error": data_result["error"]}
                else:
                    st.success("✅ Step 2: Data fetching completed!")
                    if data_result.get("data"):
                        data_keys = list(data_result["data"].keys())
                        st.write(f"Fetched {len(data_keys)} data table(s): {', '.join(data_keys)}")
            
            progress_bar.progress(60)
            
            # Check for interrupt
            check_interrupt_point("Data Fetching", {
                "query": data_result.get("query", ""),
                "data": data_result.get("data", {}),
                "feature_description": feature_description,
                "next_step": "Feature Building"
            })
            
        except PipelineInterrupt as e:
            st.info(f"⏸️ Pipeline paused at: {e.message}")
            return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
        
        # Step 3: Feature Building with real-time code updates
        try:
            with status_placeholder.container():
                st.info("🔄 Step 3: Building features with iterative debugging...")
            st.session_state.current_step = "Feature Building"
            
            # Use modified feature build that updates UI in real-time
            feature_result = run_feature_build_with_ui_updates_and_interrupts(
                query=data_result.get("query", ""),
                feature_description=feature_description,
                data=data_result.get("data", {}),
                status_placeholder=status_placeholder,
                progress_bar=progress_bar
            )
            
            if feature_result.get("paused"):
                return feature_result
            
            if feature_result.get("success"):
                add_step_result("Final Execution", {
                    "feature_table": feature_result["feature_table"],
                    "final_code": feature_result.get("final_code", "")
                })
                
                st.session_state.pipeline_state['feature_table'] = feature_result["feature_table"]
                st.session_state.step_status['feature_build'] = 'completed'
                
                with status_placeholder.container():
                    st.success("🎉 Pipeline completed successfully!")
                    st.balloons()
                
                progress_bar.progress(100)
                return {
                    "success": True,
                    "feature_description": feature_description,
                    "data": data_result.get("data", {}),
                    "result": feature_result["feature_table"]
                }
            else:
                add_step_result("Final Execution", {
                    "error": feature_result.get('error_message', 'Unknown error')
                }, "failed")
                
                with status_placeholder.container():
                    st.error(f"❌ Step 3: Feature building failed: {feature_result.get('error_message', 'Unknown error')}")
                return {"success": False, "error": feature_result.get("error_message", "Feature building failed")}
                
        except PipelineInterrupt as e:
            st.info(f"⏸️ Pipeline paused at: {e.message}")
            return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
        
    except Exception as e:
        st.session_state.pipeline_running = False
        add_step_result("Pipeline Error", {"error": str(e)}, "failed")
        with status_placeholder.container():
            st.error(f"Pipeline execution failed: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        st.session_state.pipeline_running = False

def resume_from_checkpoint(status_placeholder, progress_bar):
    """Resume pipeline execution from a checkpoint"""
    checkpoint = st.session_state.current_checkpoint
    step_name = checkpoint["step_name"]
    data = checkpoint["data"]
    
    with status_placeholder.container():
        st.info(f"🔄 Resuming from checkpoint: {step_name}")
    
    # Handle user intervention if provided
    if st.session_state.user_intervention:
        intervention_type = st.session_state.intervention_type
        intervention_text = st.session_state.user_intervention
        
        add_step_result("User Intervention", {
            "type": intervention_type,
            "instruction": intervention_text,
            "checkpoint": step_name
        })
        
        # Clear intervention data
        st.session_state.user_intervention = ""
        st.session_state.intervention_type = ""
    
    # Clear checkpoint
    st.session_state.current_checkpoint = None
    
    # Resume based on where we were
    if step_name == "Demand Analysis":
        # Continue to data fetching with potentially modified requirements
        return continue_pipeline_from_step("Data Fetching", data, status_placeholder, progress_bar)
    elif step_name == "Data Fetching":
        # Continue to feature building
        return continue_pipeline_from_step("Feature Building", data, status_placeholder, progress_bar)
    elif step_name == "Feature Building":
        # Continue or restart feature building with modifications
        return continue_pipeline_from_step("Feature Building", data, status_placeholder, progress_bar)
    else:
        with status_placeholder.container():
            st.error(f"Unknown checkpoint: {step_name}")
        return {"success": False, "error": f"Unknown checkpoint: {step_name}"}

def continue_pipeline_from_step(step_name: str, data: dict, status_placeholder, progress_bar):
    """Continue pipeline execution from a specific step"""
    # This would contain the logic to continue from where we left off
    # For now, just restart the pipeline with the saved data
    with status_placeholder.container():
        st.info(f"🔄 Continuing pipeline from {step_name}...")
    
    # Implementation would depend on the specific step
    # For now, just return success to allow manual restart
    return {"success": True, "resumed": True}

def run_feature_build_with_ui_updates_and_interrupts(query: str, feature_description: str, data: Dict, status_placeholder, progress_bar):
    """Modified feature build function that updates UI in real-time with interrupt support"""
    from feature_build import (
        generate_feature_code, eval_code, debug_fix_code, execute_final_code,
        FeatureBuildState
    )
    
    # Initialize state
    state = {
        "query": query,
        "feature_description": feature_description,
        "feature_code": "",
        "data": data,
        "feature_table": None,
        "debug_code": None,
        "debug_info": None,
        "error_message": None,
        "execution_result": None,
        "current_retry": 0,
        "max_retries": 3,
        "eval_stage": 0,
        "is_successful": False,
        "debug_history": [],
        "code_validated": False,
        "code_history": []
    }
    
    try:
        # Generate initial code
        with status_placeholder.container():
            st.info("🔧 Generating initial feature code...")
        
        # Check for user intervention on code generation
        if st.session_state.user_intervention and st.session_state.intervention_type == "Modify Feature Code":
            # Apply user's code modification instructions
            modified_feature_description = f"{feature_description}\n\nUser Modification Request: {st.session_state.user_intervention}"
            state["feature_description"] = modified_feature_description
            add_step_result("User Code Modification", {
                "original_description": feature_description,
                "modified_description": modified_feature_description,
                "user_instruction": st.session_state.user_intervention
            })
            st.session_state.user_intervention = ""
            st.session_state.intervention_type = ""
        
        code_result = generate_feature_code(state)
        state.update(code_result)
        
        # Add to results history and UI
        add_step_result("Code Generation", {
            "code": state["feature_code"],
            "retry": 0
        })
        
        st.session_state.code_history.append({
            "code": state["feature_code"],
            "status": "generated",
            "retry": 0,
            "debug_info": None
        })
        
        with status_placeholder.container():
            st.success("✅ Initial code generated!")
            with st.expander("View Generated Code"):
                st.code(state["feature_code"], language='python')
        
        progress_bar.progress(70)
        
        # Check for interrupt after code generation
        try:
            check_interrupt_point("Code Generation", {
                "feature_code": state["feature_code"],
                "state": state,
                "next_step": "Code Evaluation"
            })
        except PipelineInterrupt as e:
            return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
        
        # Evaluation and debugging loop
        retry_count = 0
        while retry_count <= state["max_retries"]:
            with status_placeholder.container():
                if retry_count == 0:
                    st.info("🔍 Evaluating generated code...")
                else:
                    st.info(f"🔍 Evaluating code (Retry {retry_count})...")
            
            # Evaluate code
            eval_result = eval_code(state)
            state.update(eval_result)
            
            # Add evaluation to results history
            add_step_result("Code Evaluation", {
                "retry": retry_count,
                "debug_info": state.get("debug_info", {}),
                "code_validated": state.get("code_validated", False)
            })
            
            # Update code history from eval result
            if "code_history" in eval_result:
                st.session_state.code_history = eval_result["code_history"]
            
            # Check for interrupt after evaluation
            try:
                check_interrupt_point("Code Evaluation", {
                    "evaluation_result": eval_result,
                    "state": state,
                    "retry_count": retry_count,
                    "next_step": "Final Execution" if state.get("code_validated", False) else "Code Debugging"
                })
            except PipelineInterrupt as e:
                return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
            
            if state.get("code_validated", False):
                # Code is valid, execute final version
                with status_placeholder.container():
                    st.success("✅ Code validation successful! Executing final version...")
                
                progress_bar.progress(90)
                
                final_result = execute_final_code(state)
                state.update(final_result)
                
                if state.get("is_successful", False):
                    progress_bar.progress(100)
                    return {
                        "success": True,
                        "feature_table": state["feature_table"],
                        "final_code": state["feature_code"],
                        "retries_used": retry_count
                    }
                else:
                    return {
                        "success": False,
                        "error_message": state.get("error_message", "Final execution failed")
                    }
            
            else:
                # Code needs debugging
                if retry_count >= state["max_retries"]:
                    with status_placeholder.container():
                        st.error(f"❌ Max retries ({state['max_retries']}) reached. Code debugging failed.")
                    break
                
                with status_placeholder.container():
                    st.warning(f"⚠️ Code validation failed. Attempting to fix (Retry {retry_count + 1})...")
                    debug_info = state.get("debug_info", {})
                    if debug_info.get("errors"):
                        with st.expander("View Debug Errors"):
                            for error in debug_info["errors"]:
                                st.error(error)
                
                # Check for user intervention on debugging
                debug_instruction = ""
                if st.session_state.user_intervention and st.session_state.intervention_type == "Modify Feature Code":
                    debug_instruction = f"\n\nUser Debug Instruction: {st.session_state.user_intervention}"
                    st.session_state.user_intervention = ""
                    st.session_state.intervention_type = ""
                
                # Debug and fix code
                debug_result = debug_fix_code(state)
                state.update(debug_result)
                
                # If user provided debug instruction, incorporate it
                if debug_instruction:
                    # This would need to be implemented in the debug_fix_code function
                    # For now, just add to debug history
                    add_step_result("User Debug Intervention", {
                        "instruction": debug_instruction,
                        "retry": retry_count + 1
                    })
                
                # Add fixed code to results history
                add_step_result("Code Fix", {
                    "fixed_code": state["feature_code"],
                    "retry": retry_count + 1,
                    "debug_instruction": debug_instruction
                })
                
                # Update UI with new code
                st.session_state.code_history.append({
                    "code": state["feature_code"],
                    "status": "debugging",
                    "retry": retry_count + 1,
                    "debug_info": state.get("debug_info")
                })
                
                with status_placeholder.container():
                    st.info(f"🔧 Generated fixed code for retry {retry_count + 1}")
                    with st.expander("View Fixed Code"):
                        st.code(state["feature_code"], language='python')
                
                retry_count += 1
                progress_bar.progress(70 + (retry_count * 5))
                
                # Check for interrupt after debugging
                try:
                    check_interrupt_point("Code Debugging", {
                        "fixed_code": state["feature_code"],
                        "state": state,
                        "retry_count": retry_count,
                        "next_step": "Code Evaluation"
                    })
                except PipelineInterrupt as e:
                    return {"success": False, "paused": True, "checkpoint": e.checkpoint_data}
        
        # If we get here, all retries failed
        return {
            "success": False,
            "error_message": f"Feature building failed after {state['max_retries']} retries",
            "debug_history": state.get("debug_history", []),
            "retries_used": retry_count
        }
        
    except Exception as e:
        add_step_result("Feature Building Error", {"error": str(e)}, "failed")
        return {
            "success": False,
            "error_message": f"Feature building error: {str(e)}"
        }

def run_pipeline_async(human_input: str):
    """Wrapper function that calls the real-time pipeline"""
    return run_pipeline_with_realtime_updates(human_input)

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📈 Quant Factor Pipeline Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for input and controls
    with st.sidebar:
        st.markdown("### 📝 Factor Request")
        
        # User input
        human_input = st.text_area(
            "Describe your desired factor:",
            height=150,
            placeholder="e.g., Create a momentum factor based on 5-minute BTCUSDT price changes with 20-period rolling average..."
        )
        
        # Pipeline controls
        display_pipeline_controls()
        
        # Run button
        if not st.session_state.pipeline_running and not st.session_state.pipeline_paused:
            run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
        else:
            run_button = False
        
        # Clear button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.pipeline_state = {}
            st.session_state.pipeline_results = None
            st.session_state.code_history = []
            st.session_state.query_info = ""
            st.session_state.current_step = ""
            st.session_state.step_status = {}
            st.session_state.step_results_history = []
            st.session_state.interrupt_requested = False
            st.session_state.current_checkpoint = None
            st.session_state.user_intervention = ""
            st.session_state.intervention_type = ""
            st.session_state.pipeline_running = False
            st.session_state.pipeline_paused = False
            st.rerun()
        
        # Display pipeline status
        display_pipeline_status(st.session_state.pipeline_state)
        
        # Show current step if pipeline is running
        if st.session_state.pipeline_running and st.session_state.current_step:
            st.markdown("### 🔄 Current Step")
            st.info(st.session_state.current_step)
        
        # Show pause status
        if st.session_state.pipeline_paused:
            st.markdown("### ⏸️ Pipeline Paused")
            st.warning("Pipeline is paused. Use controls above to resume or provide intervention.")
    
    # User intervention panel (if paused)
    if st.session_state.pipeline_paused:
        display_intervention_panel()
    
    # Run pipeline when button is clicked
    if run_button and human_input.strip():
        # Clear previous results
        st.session_state.pipeline_state = {}
        st.session_state.pipeline_results = None
        st.session_state.code_history = []
        st.session_state.query_info = ""
        st.session_state.step_results_history = []
        
        # Create a container for the real-time pipeline execution
        pipeline_container = st.container()
        
        with pipeline_container:
            st.markdown("### 🚀 Pipeline Execution")
            
            # This will be populated by the pipeline execution
            execution_placeholder = st.empty()
            
            # Run pipeline with real-time updates
            with execution_placeholder.container():
                results = run_pipeline_async(human_input.strip())
                
                if results and results.get("success"):
                    st.success("🎉 Pipeline completed successfully!")
                elif results and results.get("paused"):
                    st.info("⏸️ Pipeline paused. Use the intervention panel above to continue.")
                elif results:
                    st.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
                else:
                    st.error("❌ Pipeline execution failed")
        
        # Force a rerun to show the updated results
        if results and (results.get("success") or results.get("paused")):
            time.sleep(1)  # Brief pause to let users see the message
            st.rerun()
            
    elif run_button and not human_input.strip():
        st.error("Please enter a factor description first.")
    
    # Main content area - show results if available
    if (st.session_state.pipeline_state or st.session_state.query_info or 
        st.session_state.code_history or st.session_state.step_results_history):
        
        st.markdown("---")
        st.markdown("## 📊 Pipeline Results")
        
        # Create tabs for different result sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📜 Execution History", 
            "🔍 Query", 
            "📊 Data", 
            "💻 Code Evolution", 
            "📈 Feature Table"
        ])
        
        with tab1:
            # Execution History section (NEW - shows all step results)
            display_step_results_history()
        
        with tab2:
            # Query section
            display_query_section()
        
        with tab3:
            # KDB Data section
            display_kdb_data_section(st.session_state.pipeline_state.get('data', {}))
            
        with tab4:
            # Feature Code Evolution section
            display_code_evolution_section(st.session_state.code_history)
        
        with tab5:
            # Feature Table section
            display_feature_table_section(st.session_state.pipeline_state.get('feature_table'))
        
        # Show detailed step status at the bottom
        if st.session_state.step_status:
            with st.expander("📋 Current Pipeline Status"):
                for step, status in st.session_state.step_status.items():
                    if status == 'completed':
                        st.success(f"✅ {step}: Completed")
                    elif status == 'running':
                        st.info(f"🔄 {step}: Running...")
                    elif status == 'failed':
                        st.error(f"❌ {step}: Failed")
                    else:
                        st.warning(f"⏳ {step}: {status}")
        
        # Show checkpoint info if paused
        if st.session_state.current_checkpoint:
            with st.expander("🔍 Checkpoint Information"):
                checkpoint = st.session_state.current_checkpoint
                st.write(f"**Paused at**: {checkpoint['step_name']}")
                st.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp']))}")
                st.json(checkpoint['data'])
        
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Quant Factor Pipeline Dashboard! 👋
        
        This interactive dashboard allows you to:
        
        1. **📝 Input Factor Requests**: Describe your desired quantitative factor in natural language
        2. **🔍 View Query**: See the generated query for data fetching
        3. **📊 View Data**: Explore the fetched market data with interactive tables
        4. **💻 Track Code Evolution**: See how the feature generation code evolves through debugging cycles
        5. **📈 Analyze Results**: Examine the final feature table with statistics and insights
        6. **⏸️ Interactive Control**: Pause, resume, and intervene in the pipeline execution
        
        **To get started:**
        1. Enter your factor description in the sidebar
        2. Click "🚀 Run Pipeline"
        3. Watch the real-time execution! ✨
        4. Use ⏸️ Pause button to intervene when needed
        
        ---
        
        ### Example Factor Requests:
        
        ```
        Create a momentum factor based on BTCUSDT 5-minute price changes, 
        smoothed with a 20-period rolling average
        ```
        
        ```
        Build a volatility factor using ETHUSDT hourly returns 
        with 14-period standard deviation
        ```
        
        ### Interactive Features:
        
        - ⚡ **Live Progress**: Watch each step execute in real-time
        - 🔄 **Code Evolution**: See code generation and debugging happen live
        - 📊 **Instant Data**: View fetched data as soon as it's available
        - 🎯 **Step-by-step**: Clear visibility into each pipeline stage
        - ⏸️ **Pause & Resume**: Interrupt execution to provide guidance
        - 🛠️ **User Intervention**: Modify code, queries, or parameters mid-execution
        - 📜 **Complete History**: All step results are preserved and viewable
        """)

if __name__ == "__main__":
    main() 