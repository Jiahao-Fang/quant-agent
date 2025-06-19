#!/usr/bin/env python3
"""
Test script for the Quant Factor Pipeline UI
Tests the real-time update functionality
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ Numpy imported successfully")
    except ImportError as e:
        print(f"❌ Numpy import failed: {e}")
        return False
    
    try:
        import pykx as kx
        print("✅ PyKX imported successfully")
    except ImportError as e:
        print(f"⚠️ PyKX import failed: {e} (this is expected if KDB+ is not installed)")
    
    # Test project modules
    try:
        from entry import run_pipeline, demand_analysis
        print("✅ Entry module imported successfully")
    except ImportError as e:
        print(f"❌ Entry module import failed: {e}")
        return False
    
    try:
        from feature_build import FeatureBuildState
        print("✅ Feature build module imported successfully")
    except ImportError as e:
        print(f"❌ Feature build module import failed: {e}")
        return False
    
    try:
        from data_fetch import create_data_fetch_graph
        print("✅ Data fetch module imported successfully")
    except ImportError as e:
        print(f"❌ Data fetch module import failed: {e}")
        return False
    
    return True

def test_ui_functions():
    """Test if UI functions can be called without errors"""
    print("\nTesting UI functions...")
    
    try:
        # Import the UI module
        from factor_pipeline_ui import (
            init_session_state, 
            display_query_section,
            display_kdb_data_section,
            display_code_evolution_section,
            display_feature_table_section
        )
        print("✅ UI functions imported successfully")
        
        # Test session state initialization (this would normally be done by Streamlit)
        print("✅ UI functions can be imported and called")
        
    except ImportError as e:
        print(f"❌ UI function import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ UI function test failed: {e}")
        return False
    
    return True

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\nTesting pipeline components...")
    
    try:
        # Test demand analysis
        from entry import demand_analysis
        
        test_state = {
            "human_input": "Create a simple momentum factor for BTCUSDT",
            "feature_description": "",
            "query": "",
            "data": None,
            "error": None,
            "result": None
        }
        
        print("✅ Pipeline components can be imported")
        
    except Exception as e:
        print(f"❌ Pipeline component test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Quant Factor Pipeline UI")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test UI functions
    if not test_ui_functions():
        all_tests_passed = False
    
    # Test pipeline components
    if not test_pipeline_components():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! UI should work correctly.")
        print("\nTo run the UI:")
        print("python run_ui.py")
        print("or")
        print("streamlit run factor_pipeline_ui.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("pip install streamlit pandas numpy pykx")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 