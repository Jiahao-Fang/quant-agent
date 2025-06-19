"""
Test cases for feature building functionality.
"""

import pytest
import pykx as kx
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_build import (
    FeatureBuildState,
    generate_feature_code,
    eval_code,
    debug_fix_code,
    execute_final_code,
    create_feature_build_graph,
    run_feature_build
)

@pytest.fixture
def sample_feature_description():
    return "**One-Sentence Summary:**   \nThe manager wants to construct a momentum factor based on the rate of change of the BTCUSDT price, using 5-minute K-line data and smoothing it with a rolling average over 20 data points. \n\n**Required Fields:**  \nThe fields required from the available dataset are `symbol` and `tradeTimestamp` from the `trade` updates, and `price` for the calculation of the rate of change. \n\n**Step-By-Step Outline To Compute The Factor:**   \n  \n1. Filter data to include only BTCUSDT's data using the `symbol` field.\n2. Resample the `price` data with 5-minute intervals using the `tradeTimestamp`.\n3. Calculate the rate of change for the price by taking the difference between the current price and the price at the previous time point, then divide it by the price at the previous time point.\n4. Smooth the rate of change using a rolling mean (moving average) over 20 data points.\n5. Store the results in a new dataset.\n\n**Special Considerations:**   \n- Ensure that data is in the right timezone, and if not - convert it to the desired timezone.\n- Handle missing values appropriately - consider forward filling or backward filling to fill gaps in data, but avoid extrapolation.\n- Ensure that data is correctly resampled - check if start and end times align with full 5-minute intervals.\n- Verify liquidity by checking if there is enough trade volume - this may be done by either filtering on `quantity` or creating a separate filter based on trading volume."


@pytest.fixture
def sample_query():
    return '''[{
        "symbol": "BTCUSDT",
        "exchange": "ST_BNS",
        "data_sources": [{
            "type": "trade",
            "required_fields": ["eventTimestamp", "price", "quantity"]
        }]
    }]'''

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

# def test_generate_feature_code(sample_feature_description, sample_query):
#     """Test feature code generation"""
#     state = {
#         "feature_description": sample_feature_description,
#         "query": sample_query
#     }
    
#     result = generate_feature_code(state)
    
#     assert "feature_code" in result
#     assert isinstance(result["feature_code"], str)
#     assert "def compute_factor" in result["feature_code"]
#     assert result["current_retry"] == 0
#     assert not result["is_successful"]

# def test_eval_code_success(sample_feature_description, sample_query, sample_data_dict):
#     """Test successful code evaluation"""
#     # First generate the code
#     state = {
#         "feature_description": sample_feature_description,
#         "query": sample_query,
#         "feature_code": """
# import pykx as kx
# from typing import Dict

# def compute_factor(data_dict: Dict[str, kx.Table]) -> kx.Table:
#     trades = data_dict["ST_BNS_BTCUSDT_trade"]
#     price = trades['price']
#     result = kx.Table({'momentum': price})
#     return result
# """,
#         "data": sample_data_dict
#     }
    
#     result = eval_code(state)
    
#     assert "debug_code" in result
#     assert result.get("debug_error") is None
#     assert isinstance(result.get("debug_result"), dict)



def test_debug_fix_code(sample_feature_description, sample_query):
    """Test code debugging and fixing"""
    state = {
        "feature_description": sample_feature_description,
        "query": sample_query,
        "feature_code": "def compute_factor(data_dict): return None  # Incomplete code",
        "debug_result": {
            "errors": ["Invalid return type"],
            "variables": {},
            "steps": ["Step 1: Data extraction"]
        },
        "current_retry": 0,
        "max_retries": 3
    }
    
    result = debug_fix_code(state)
    
    assert "feature_code" in result
    assert isinstance(result["feature_code"], str)
    assert result["current_retry"] == 1
    assert not result.get("code_validated", False)

def test_execute_final_code(sample_data_dict):
    """Test final code execution"""
    state = {
        "feature_code": """
import pykx as kx
from typing import Dict

def compute_factor(data_dict: Dict[str, kx.Table]) -> kx.Table:
    trades = data_dict["ST_BNS_BTCUSDT_trade"]
    price = trades['price']
    result = kx.Table(data = {'momentum': price})
    return result
result = compute_factor(data_dict)
""",
        "data": sample_data_dict
    }
    
    result = execute_final_code(state)
    
    assert result["is_successful"]
    assert result["feature_table"] is not None
    assert isinstance(result["feature_table"], kx.Table)
    assert result["error_message"] is None

def test_feature_build_graph():
    """Test the feature building graph creation and basic execution"""
    graph = create_feature_build_graph()
    
    initial_state = {
        "query": "sample query",
        "feature_description": "sample description",
        "feature_code": "",
        "data": {},
        "feature_table": None,
        "current_retry": 0,
        "max_retries": 3,
        "eval_stage": 0,
        "is_successful": False,
        "debug_history": [],
        "code_validated": False
    }
    
    # Execute the graph
    try:
        final_state = graph.invoke(initial_state)
        assert isinstance(final_state, dict)
    except Exception as e:
        pytest.fail(f"Graph execution failed: {str(e)}")

def test_run_feature_build(sample_query, sample_feature_description, sample_data_dict):
    """Test the complete feature building process"""
    result = run_feature_build(
        query=sample_query,
        feature_description=sample_feature_description,
        data=sample_data_dict
    )
    
    assert isinstance(result, dict)
    assert "success" in result
    if result["success"]:
        assert "feature_table" in result
        assert isinstance(result["feature_table"], kx.Table)
    else:
        assert "error_message" in result
        assert "debug_history" in result 

def test_eval_code_failure(sample_feature_description, sample_query, sample_data_dict):
    """Test code evaluation with invalid code that should fail"""
    # 准备一个包含错误的状态
    state = {
        "feature_description": sample_feature_description,
        "query": sample_query,
        "feature_code": """
import pykx as kx
from typing import Dict

def compute_factor(data_dict: Dict[str, kx.Table]) -> kx.Table:
    # 故意引入错误：
    # 1. 使用未定义的变量
    # 2. 错误的字典键名
    trades = data_dict["WRONG_KEY"]  # 错误的键名
    undefined_variable = some_undefined_variable  # 未定义的变量
    result = kx.Table({'momentum': undefined_variable})
    return result
""",
        "data": sample_data_dict,
        "current_retry": 0,
        "code_history": []
    }
    
    result = eval_code(state)
    
    # 验证错误处理
    assert "debug_code" in result
    assert "debug_info" in result
    debug_info = result["debug_info"]
    
    # 验证debug_info结构
    assert isinstance(debug_info, dict)
    assert "steps" in debug_info
    assert "variables" in debug_info
    assert "errors" in debug_info
    assert "success" in debug_info
    assert "final_result" in debug_info
    
    # 验证错误状态
    assert debug_info["success"] is False  # 应该失败
    assert len(debug_info["errors"]) > 0  # 应该有错误信息
    assert not result.get("code_validated", True)  # 代码验证应该失败
    assert not result.get("is_successful", True)  # 整体应该失败
    
    # 验证代码历史追踪
    assert "code_history" in result
    code_history = result["code_history"]
    assert len(code_history) > 0  # 应该记录了这次失败
    assert code_history[-1]["status"] == "failed"  # 最新记录应该是失败状态 