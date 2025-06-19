"""
DataFetcher Processor - AI-driven data fetcher that generates optimized KDB+ queries and retrieves data.
"""

from typing import Dict, Any, List, Optional, TypedDict
import logging
import json
import datetime
import pykx as kx
from langchain_openai import ChatOpenAI

from ..core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from ..core.decorators import evaluable, debuggable, observable, interruptible
from ..prompt_lib import get_prompt_manager

logger = logging.getLogger(__name__)

# State type for data fetching stage
class DataFetcherState(TypedDict):
    feature_description: str  # Feature description to generate query from
    query: str  # Generated query in JSON string format containing data request parameters
    data: Optional[Dict[str, kx.Table]]  # Fetched data
    error: Optional[str]  # Error message if any

@observable(observers=["ui", "logger"])
@evaluable(max_retries=2)
@debuggable(max_retries=1)
@interruptible(save_point_id="data_fetch")
class DataFetcher(BaseProcessor):
    """
    AI-driven data fetcher that generates optimized KDB+ queries and retrieves data.
    
    Capabilities:
    - Observable: UI monitoring and logging
    - Evaluable: Data quality validation
    - Debuggable: Query error handling and retry
    - Interruptible: User can pause/resume data fetching
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataFetcher.
        
        Required config:
        - db_path: KDB database path (optional, defaults to 'D:/kdbdb')
        
        Optional config:
        - prompt_manager: Custom prompt manager (defaults to global instance)
        - model_name: LLM model name (defaults to 'gpt-4')
        """
        super().__init__(config)
        
        # Extract dependencies from config
        self.db_path = config.get('db_path', 'D:/kdbdb')
        self.model_name = config.get('model_name', 'gpt-4')
        
        # Use provided prompt manager or get global instance
        self.prompt_manager = config.get('prompt_manager', get_prompt_manager())
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.model_name)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def get_processor_type(self) -> ProcessorType:
        """Return processor type."""
        return ProcessorType.DATA_FETCHER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """
        Core data fetching logic: generate query and fetch data.
        
        Args:
            state: Input contains feature_description
            
        Returns:
            Updated state with query and fetched data
        """
        try:
            # Extract feature description
            input_data = state['input_data']
            if not isinstance(input_data, dict) or 'feature_description' not in input_data:
                raise ValueError("Input must contain 'feature_description' field")
            
            feature_description = input_data['feature_description']
            self.logger.info(f"Processing feature description: {feature_description[:100]}...")
            
            # Step 1: Generate query using AI
            query_json = self._generate_kdb_query(feature_description)
            self.logger.info("Successfully generated KDB query")
            
            # Step 2: Execute query and fetch data
            data_tables = self._fetch_kdb_data(query_json)
            self.logger.info(f"Successfully fetched {len(data_tables)} data tables")
            
            # Update state with results
            state['output_data'] = {
                'query': query_json,
                'data': data_tables,
                'feature_description': feature_description
            }
            state['status'] = 'success'
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data fetching failed: {e}")
            state['error'] = str(e)
            state['status'] = 'error'
            raise
    
    def _generate_kdb_query(self, feature_description: str) -> str:
        """
        Generate KDB+ query using AI based on feature description.
        
        Args:
            feature_description: Natural language description of desired features
            
        Returns:
            JSON string containing KDB query specification
        """
        try:
            # Get the appropriate prompt template
            prompt_content = self.prompt_manager.format_template(
                processor_type='data_fetcher',
                template_name='data_fetcher_lead',
                feature_description=feature_description
            )
            
            if prompt_content is None:
                # Fallback to a basic prompt if template not found
                prompt_content = f"""Generate a KDB+ query for the following feature description:{feature_description}
                Return a JSON object with query specifications."""
                self.logger.warning("Using fallback prompt - data_fetcher_lead template not found")
            
            # Generate response using LLM
            response = self.llm.invoke(prompt_content)
            query_json = response.content.strip()
            
            # Extract JSON content from response if needed
            if "```json" in query_json:
                start = query_json.find("```json") + 7
                end = query_json.find("```", start)
                query_json = query_json[start:end].strip()
            
            # Remove literal \n characters from response
            query_json = query_json.replace('\\n', '')
            
            # Validate JSON format
            try:
                json.loads(query_json)
            except json.JSONDecodeError:
                raise ValueError("Generated query is not valid JSON")
            
            return query_json
            
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            raise
    
    def _fetch_kdb_data(self, input_json: str) -> Dict[str, kx.Table]:
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
        db = kx.DB(path=self.db_path)
        
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
                    self.logger.warning(f"跳过无效的数据类型: {data_type}")
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
                    
                    self.logger.info(f"成功获取 {dict_key} 数据")
                    
                except AttributeError:
                    self.logger.error(f"表 {data_type} 不存在")
                except Exception as e:
                    self.logger.error(f"查询 {data_type} 时出错: {str(e)}")
        
        return result
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """
        Evaluate data quality and completeness using pykx methods.
        
        Args:
            state: State with fetched data
            
        Returns:
            Updated state with evaluation results
        """
        try:
            output_data = state.get('output_data')
            if not output_data:
                state['eval_passed'] = False
                state['eval_reason'] = "No output data to evaluate"
                return state
            
            data_tables = output_data.get('data', {})
            
            # Check 1: Data exists
            if not data_tables:
                state['eval_passed'] = False
                state['eval_reason'] = "No data tables returned"
                return state
            
            # Check 2: Tables have minimum row count using pykx methods
            min_rows = self.config.get('min_rows_per_table', 100)
            total_rows = 0
            quality_issues = []
            
            for table_name, table_data in data_tables.items():
                if isinstance(table_data, kx.Table):
                    # Use pykx's count function to get number of rows
                    table_rows = kx.q.count(table_data)
                    total_rows += table_rows
                    self.logger.info(f"Table {table_name}: {table_rows} rows")
                    
                    # Get table metadata
                    meta = kx.q.meta(table_data)
                    self.logger.info(f"Table {table_name} metadata: {meta}")
                    
                    # Check for null values in each column
                    for col in meta['c']:
                        null_count = kx.q.sum(kx.q.null table_data[col])
                        if null_count > 0:
                            quality_issues.append(f"Column {col} in {table_name} has {null_count} null values")
                    
                    # Check for data type consistency
                    for col in meta['c']:
                        col_type = meta[meta['c'] == col]['t'][0]
                        if col_type in ['f', 'e']:  # float or real
                            # Check for infinity values
                            inf_count = kx.q.sum(kx.q.abs(table_data[col]) == kx.q.inf)
                            if inf_count > 0:
                                quality_issues.append(f"Column {col} in {table_name} has {inf_count} infinity values")
                    # Check for time series continuity if date/time column exists
                    if 'date' in meta['c']:
                        dates = kx.q.asc(kx.q.unique(table_data['date']))
                        if kx.q.count(dates) > 1:
                            gaps = kx.q.deltas(dates)
                            if kx.q.any(gaps > 1):
                                quality_issues.append(f"Table {table_name} has gaps in date sequence")
            if total_rows < min_rows:
                state['eval_passed'] = False
                state['eval_reason'] = f"Insufficient data: {total_rows} rows < {min_rows} required"
                return state
            
            # Check 3: Query validity
            query = output_data.get('query')
            if not query or not self._validate_query_format(query):
                state['eval_passed'] = False
                state['eval_reason'] = "Invalid query format"
                return state
            
            # All checks passed
            state['eval_passed'] = True
            state['eval_reason'] = f"Data quality validated: {total_rows} total rows across {len(data_tables)} tables"
            
            # Add quality issues to metadata if any
            if quality_issues:
                state['metadata'] = state.get('metadata', {})
                state['metadata']['quality_issues'] = quality_issues
                self.logger.warning(f"Data quality issues found: {quality_issues}")
            
            self.logger.info(f"Data evaluation passed: {state['eval_reason']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Data evaluation failed: {e}")
            state['eval_passed'] = False
            state['eval_reason'] = f"Evaluation error: {str(e)}"
            return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """
        Debug data fetching errors and determine retry strategy.
        
        Args:
            state: State with error information
            
        Returns:
            Updated state with debug analysis and retry decision
        """
        error = state.get('error')
        if not error:
            state['should_retry'] = False
            return state
        
        error_str = str(error).lower()
        
        # Analyze error type and determine retry strategy
        if 'connection' in error_str or 'timeout' in error_str:
            # Network/connection issues - worth retrying
            state['should_retry'] = True
            state['debug_reason'] = "Connection error detected - will retry"
            self.logger.info("Debug: Connection error - scheduling retry")
            
        elif 'json' in error_str or 'parse' in error_str:
            # JSON parsing errors - may be transient AI issue
            state['should_retry'] = True
            state['debug_reason'] = "Query generation error - will retry with different prompt"
            self.logger.info("Debug: Query format error - scheduling retry")
            
        elif 'permission' in error_str or 'access' in error_str:
            # Permission errors - unlikely to resolve with retry
            state['should_retry'] = False
            state['debug_reason'] = "Access permission error - manual intervention required"
            self.logger.warning("Debug: Permission error - manual fix needed")
            
        elif 'symbol' in error_str or 'not found' in error_str:
            # Data not found - may need different query parameters
            state['should_retry'] = False
            state['debug_reason'] = "Data not found - check symbol/date parameters"
            self.logger.warning("Debug: Data not found - check parameters")
            
        else:
            # Unknown error - try once more
            state['should_retry'] = True
            state['debug_reason'] = f"Unknown error: {error_str[:100]} - will retry once"
            self.logger.warning(f"Debug: Unknown error - {error_str[:100]}")
        
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """
        Handle user interrupt requests during data fetching.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated state with interrupt handling
        """
        try:
            # Save current progress
            progress = state.get('metadata', {}).get('progress', 0.0)
            
            # Update state to paused
            state['status'] = 'paused'
            state['interrupt_reason'] = f"User requested pause at {progress:.1%} completion"
            
            # Estimate completion time if available
            if 'start_time' in state.get('metadata', {}):
                elapsed = datetime.datetime.now() - state['metadata']['start_time']
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    state['completion_estimate'] = f"Estimated {remaining.total_seconds():.0f}s remaining"
            
            self.logger.info(f"Data fetching paused: {state['interrupt_reason']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Error handling interrupt: {e}")
            state['interrupt_reason'] = f"Interrupt handling error: {str(e)}"
            return state
    
    def _validate_query_format(self, query_json: str) -> bool:
        """
        Validate that query JSON has proper format.
        
        Args:
            query_json: JSON string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            query_data = json.loads(query_json)
            
            # Basic validation - should be a list or dict
            if not isinstance(query_data, (list, dict)):
                return False
            
            # Additional validation could be added here
            return True
            
        except (json.JSONDecodeError, TypeError):
            return False 