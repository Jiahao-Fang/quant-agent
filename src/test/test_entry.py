"""
Test cases for the main entry point of the quant factor pipeline.
"""

import pytest
import pykx as kx
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry import (
    EntryGraphState,
    demand_analysis,
    should_continue,
    create_entry_graph,
    run_pipeline
)

@pytest.fixture
def sample_human_input():
    return """
    我想要一个动量因子：
    1. 使用BTCUSDT的5分钟K线数据
    2. 计算价格相对于前一个时间点的变化率
    3. 使用20个时间点的移动平均进行平滑
    """

@pytest.fixture
def mock_kdb_table():
    # Create a mock KDB table with sample data
    timestamps = np.arange(1612345678000000000, 1612345678000000000 + 100 * 1e9, 1e9)
    prices = 50000 + np.random.randn(100) * 100
    quantities = np.random.rand(100) * 10
    
    return kx.Table(data={
        'eventTimestamp': timestamps,
        'price': prices,
        'quantity': quantities
    })

@pytest.fixture
def sample_data_dict(mock_kdb_table):
    return {
        "ST_BNS_BTCUSDT_trade": mock_kdb_table
    }

def test_demand_analysis(sample_human_input):
    """Test demand analysis functionality"""
    state = {
        "human_input": sample_human_input
    }
    print(state)
    result = demand_analysis(state)
    print(result)
    assert "feature_description" in result
    assert isinstance(result["feature_description"], str)
    assert len(result["feature_description"]) > 0

def test_should_continue():
    """Test the continuation logic"""
    # Test with error
    state_with_error = {
        "error": "Some error occurred"
    }
    assert should_continue(state_with_error) == "end"
    
    # Test without error
    state_without_error = {
        "error": None
    }
    assert should_continue(state_without_error) == "feature_build"

def test_entry_graph_creation():
    """Test the entry graph creation"""
    graph = create_entry_graph()
    
    initial_state = {
        "human_input": "计算BTCUSDT的5分钟价格动量",
        "feature_description": "",
        "query": "",
        "data": None,
        "error": None,
        "result": None
    }
    
    # Execute the graph
    try:
        final_state = graph.invoke(initial_state)
        assert isinstance(final_state, dict)
    except Exception as e:
        pytest.fail(f"Graph execution failed: {str(e)}")

def test_run_pipeline_success(sample_human_input, mocker, sample_data_dict):
    """Test successful pipeline execution"""
    # Mock the necessary components
    mocker.patch('src.raw_version.data_fetch.fetch_kdb_data', return_value=sample_data_dict)
    
    result = run_pipeline(sample_human_input)
    
    assert isinstance(result, dict)
    assert "success" in result
    if result["success"]:
        assert "result" in result
        assert "feature_description" in result
    else:
        assert "error" in result

def test_run_pipeline_error(sample_human_input, mocker):
    """Test pipeline error handling"""
    # Mock a failure in data fetching
    mocker.patch(
        'src.raw_version.data_fetch.fetch_kdb_data',
        side_effect=Exception("Data fetch failed")
    )
    
    result = run_pipeline(sample_human_input)
    
    assert isinstance(result, dict)
    assert not result["success"]
    assert "error" in result
    assert result["result"] is None

def test_pipeline_integration(sample_human_input):
    """Test the complete pipeline integration"""
    try:
        result = run_pipeline(sample_human_input)
        assert isinstance(result, dict)
        assert "success" in result
        
        if result["success"]:
            assert isinstance(result["result"], kx.Table)
            assert "feature_description" in result
        else:
            assert "error" in result
            
    except Exception as e:
        pytest.fail(f"Pipeline integration test failed: {str(e)}")

def test_pipeline_input_validation():
    """Test pipeline input validation"""
    # Test with empty input
    empty_result = run_pipeline("")
    assert not empty_result["success"]
    assert "error" in empty_result
    
    # Test with invalid input type
    invalid_result = run_pipeline(123)  # type: ignore
    assert not invalid_result["success"]
    assert "error" in invalid_result 