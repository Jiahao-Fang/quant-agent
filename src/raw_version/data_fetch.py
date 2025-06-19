"""
Data fetching module for the quant factor pipeline.
Handles data querying and fetching from KDB+ database.
"""

from typing import Dict, TypedDict, Optional
import pykx as kx
import datetime
import json
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import templates from prompt_lib
from prompt_lib.data_fetcher_lead import DATA_FETCHER_LEAD_TEMPLATE
from prompt_lib.data_fetcher_dev import DATA_FETCHER_DEV_TEMPLATE

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# State type for data fetching stage
class DataFetcherState(TypedDict):
    feature_description: str  # Feature description to generate query from
    query: str  # Generated query in JSON string format containing data request parameters
    data: Optional[Dict[str, kx.Table]]  # Fetched data
    error: Optional[str]  # Error message if any

def generate_query(state: DataFetcherState) -> Dict[str, str]:
    """Generate data query from feature description"""
    prompt = DATA_FETCHER_LEAD_TEMPLATE.format(feature_description=state["feature_description"])
    response = llm.invoke(prompt).content
    
    # Extract JSON content from response if needed
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    # Remove literal \n characters from response
    response = response.replace('\\n', '')
    # Parse JSON to validate format
    try:
        json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Generated query is not valid JSON")
        
    return {"query": response}

def fetch_kdb_data(input_json: str) -> Dict[str, kx.Table]:
    """
    从KDB+数据库获取数据并返回以 "exchange_symbol_datatype" 为键的一级字典
    
    Args:
        input_json: JSON字符串，包含查询请求的列表，每个请求包含:
            - symbol: 交易对，如 "BTCUSDT"
            - exchange: 交易所，如 "ST_BNS"
            - start_date: 开始日期，格式为 "YYYY-MM-DD"
            - end_date: 结束日期，格式为 "YYYY-MM-DD"
            - data_sources: 列表，每项包含:
                - type: 数据类型，如 "partial20", "BBO", "trade"
                - required_fields: 需要查询的字段列表
    
    Returns:
        Dict[str, kx.Table]: 
        一级字典，格式为 {"exchange_symbol_datatype": kx.Table}
    """
    # Extract JSON content from response if needed
    if "```json" in input_json:
        start = input_json.find("```json") + 7
        end = input_json.find("```", start)
        input_json = input_json[start:end].strip()
    
    # Parse JSON input
    input_data = json.loads(input_json)
    # 连接到KDB数据库
    db = kx.DB(path="D:/kdbdb")
    
    # 准备结果字典
    result = {}
    
    # 处理每个请求
    for request in input_data:
        symbol = request.get("symbol")
        exchange = request.get("exchange")
        start_date = request.get("start_date")
        end_date = request.get("end_date")
        data_sources = request.get("data_sources", [])
        
        # 转换日期格式为datetime对象
        if start_date:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # 为每个数据源查询数据
        for source in data_sources:
            data_type = source.get("type")
            required_fields = source.get("required_fields", [])
            
            # 确保data_type是有效的
            if not data_type or data_type not in ["partial20", "BBO", "trade"]:
                print(f"跳过无效的数据类型: {data_type}")
                continue
            
            # 创建字典键
            dict_key = f"{exchange}_{symbol}_{data_type}"
            
            # 构建查询条件
            conditions = []
            if symbol:
                conditions.append(kx.Column('symbol') == symbol)
            if exchange:
                conditions.append(kx.Column('exchange') == exchange)
            
            # 添加日期条件
            if start_date and end_date:
                if start_date == end_date:
                    conditions.append(kx.Column('date') == start_date)
                else:
                    conditions.append((kx.Column('date') >= start_date) & (kx.Column('date') <= end_date))
            
            # 组合所有条件
            where_clause = None
            for condition in conditions:
                if where_clause is None:
                    where_clause = condition
                else:
                    where_clause = where_clause & condition
            
            try:
                # 获取对应的表
                table = getattr(db, data_type)
                
                # 构建列选择
                columns = None
                for field in required_fields:
                    if columns is None:
                        columns = kx.Column(field)
                    else:
                        columns = columns & kx.Column(field)
                
                # 如果没有指定字段，获取所有字段
                if not required_fields:
                    # 执行查询，获取所有字段
                    query_result = table.select(where=where_clause)
                else:
                    # 执行查询，获取指定字段
                    query_result = table.select(columns, where=where_clause)
                
                # 存储原始kx.Table结果到一级字典
                result[dict_key] = query_result
                
                print(f"成功获取 {dict_key} 数据")
                
            except AttributeError:
                print(f"表 {data_type} 不存在")
            except Exception as e:
                print(f"查询 {data_type} 时出错: {str(e)}")
    
    return result

def execute_query(state: DataFetcherState) -> Dict:
    """Execute the generated query and fetch data"""
    try:
        data = fetch_kdb_data(state["query"])
        return {"data": data, "error": None}
    except Exception as e:
        return {"data": None, "error": str(e)}

def should_continue(state: DataFetcherState) -> str:
    """Determine if the pipeline should continue or retry"""
    if state.get("error"):
        return "end"
    return "end"

def create_data_fetch_graph() -> StateGraph:
    """
    Create and return the data fetching graph.
    
    Returns:
        StateGraph: Compiled data fetching graph
    """
    # Create graph builder
    df_builder = StateGraph(DataFetcherState)
    
    # Add nodes
    df_builder.add_node("generate_query", generate_query)
    df_builder.add_node("execute_query", execute_query)
    
    # Add edges
    df_builder.add_edge(START, "generate_query")
    df_builder.add_edge("generate_query", "execute_query")
    
    # Add conditional edges
    df_builder.add_conditional_edges(
        "execute_query",
        should_continue,
        {
            "end": END
        }
    )
    
    # Compile and return
    return df_builder.compile()
