"""
Test cases for data fetching functionality.
"""

import pytest
import json
import pykx as kx
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fetch import (
    DataFetcherState,
    generate_query,
    fetch_kdb_data,
    create_data_fetch_graph
)

@pytest.fixture
def sample_feature_description():
    return "**One-Sentence Summary:**   \nThe manager wants to construct a momentum factor based on the rate of change of the BTCUSDT price, using 5-minute K-line data and smoothing it with a rolling average over 20 data points. \n\n**Required Fields:**  \nThe fields required from the available dataset are `symbol` and `tradeTimestamp` from the `trade` updates, and `price` for the calculation of the rate of change. \n\n**Step-By-Step Outline To Compute The Factor:**   \n  \n1. Filter data to include only BTCUSDT's data using the `symbol` field.\n2. Resample the `price` data with 5-minute intervals using the `tradeTimestamp`.\n3. Calculate the rate of change for the price by taking the difference between the current price and the price at the previous time point, then divide it by the price at the previous time point.\n4. Smooth the rate of change using a rolling mean (moving average) over 20 data points.\n5. Store the results in a new dataset.\n\n**Special Considerations:**   \n- Ensure that data is in the right timezone, and if not - convert it to the desired timezone.\n- Handle missing values appropriately - consider forward filling or backward filling to fill gaps in data, but avoid extrapolation.\n- Ensure that data is correctly resampled - check if start and end times align with full 5-minute intervals.\n- Verify liquidity by checking if there is enough trade volume - this may be done by either filtering on `quantity` or creating a separate filter based on trading volume."

@pytest.fixture
def sample_query_json():
    return '''[
        {
            "symbol": "BTCUSDT",
            "exchange": "ST_BNS",
            "start_date": "2025-02-03",
            "end_date": "2025-02-03",
            "data_sources": [
                {
                    "type": "trade",
                    "required_fields": [
                        "eventTimestamp",
                        "price",
                        "quantity"
                    ]
                }
            ]
        }
    ]'''

@pytest.fixture
def mock_kdb_table():
    # Create a simple mock KDB table
    return kx.Table(data={
        'eventTimestamp': [1612345678000000000, 1612345679000000000],
        'price': [50000.0, 50100.0],
        'quantity': [1.5, 2.0]
    })

def test_generate_query(sample_feature_description):
    """Test query generation from feature description"""
    state = {
        "feature_description": sample_feature_description,
        
    }
    result = generate_query(state)
    
    assert "query" in result
    query_json = result["query"]
    # Verify the query is valid JSON
    query_data = json.loads(query_json)
    assert isinstance(query_data, list)
    assert len(query_data) > 0
    assert "symbol" in query_data[0]
    assert "exchange" in query_data[0]
    assert "data_sources" in query_data[0]

def test_fetch_kdb_data_json_extraction(sample_query_json):
    """Test JSON extraction from markdown code blocks"""
    # Test with markdown code block
    markdown_query = f"```json\n{sample_query_json}\n```"
    result = fetch_kdb_data(markdown_query)
    assert isinstance(result, dict)

def test_fetch_kdb_data_structure(sample_query_json, mocker, mock_kdb_table):
    """Test the structure of fetched data"""
    # Mock the KDB connection and query
    mocker.patch('pykx.Table.select', return_value=mock_kdb_table)
    result = fetch_kdb_data(sample_query_json)
    
    assert isinstance(result, dict)
    expected_key = "ST_BNS_BTCUSDT_trade"
    assert expected_key in result
    assert isinstance(result[expected_key], kx.Table)

def test_data_fetch_graph():
    """Test the data fetching graph creation and basic execution"""
    graph = create_data_fetch_graph()
    
    initial_state = {
        "feature_description": "计算BTCUSDT的5分钟价格动量",
        "query": "",
        "data": None,
        "error": None
    }
    
    # Execute the graph
    try:
        final_state = graph.invoke(initial_state)
        assert isinstance(final_state, dict)
        assert "query" in final_state
    except Exception as e:
        pytest.fail(f"Graph execution failed: {str(e)}")

def test_error_handling(mocker):
    """Test error handling in data fetching"""
    # Mock a database error
    mocker.patch('pykx.DB', side_effect=Exception("Database connection failed"))
    
    with pytest.raises(Exception):
        fetch_kdb_data('''[{
            "symbol": "BTCUSDT",
            "exchange": "ST_BNS",
            "start_date": "2025-02-03",
            "end_date": "2025-02-03",
            "data_sources": [{"type": "trade", "required_fields": ["price"]}]
        }]''') 