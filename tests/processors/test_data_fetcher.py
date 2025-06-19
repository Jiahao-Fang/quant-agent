"""
Tests for src/processors/data_fetcher.py

Tests DataFetcher with new decorator-based architecture:
- Observable, Evaluable, Debuggable, Interruptible capabilities
- Core data fetching logic
- KDB+ query generation and execution
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import pykx as kx

from src.processors.data_fetcher import DataFetcher
from src.core.base_processor import ProcessorType, ProcessorState


class TestDataFetcher:
    """Test DataFetcher functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'db_path': '/test/db',
            'min_rows_per_table': 50
        }
    
    def test_initialization(self):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(self.config)
        
        assert fetcher.get_processor_type() == ProcessorType.DATA_FETCHER
        assert fetcher.model_name == 'gpt-4'
        assert fetcher.prompt_manager == self.mock_prompt_manager
        assert fetcher.db_path == '/test/db'
    
    def test_initialization_missing_dependencies(self):
        """Test initialization with missing dependencies."""
        # Missing model_name
        with pytest.raises(ValueError, match="model_name is required"):
            DataFetcher({'prompt_manager': self.mock_prompt_manager})
        
        # Missing prompt_manager
        with pytest.raises(ValueError, match="prompt_manager is required"):
            DataFetcher({'model_name': 'gpt-4'})
    
    def test_capabilities(self):
        """Test processor capabilities."""
        fetcher = DataFetcher(self.config)
        capabilities = fetcher.get_capabilities()
        
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
    
    @patch('src.processors.data_fetcher.DataFetcher._execute_kdb_query')
    @patch('src.processors.data_fetcher.DataFetcher._generate_kdb_query')
    def test_process_core_logic_success(self, mock_generate, mock_execute):
        """Test successful core processing logic."""
        fetcher = DataFetcher(self.config)
        
        # Setup mocks
        mock_generate.return_value = '{"query": "select from table"}'
        mock_execute.return_value = {'table1': [{'col1': 1, 'col2': 2}]}
        
        # Create input state
        state: ProcessorState = {
            'input_data': {'feature_description': 'Test feature description'},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        # Process
        result_state = fetcher._process_core_logic(state)
        
        # Verify results
        assert result_state['status'] == 'success'
        assert 'output_data' in result_state
        
        output_data = result_state['output_data']
        assert 'query' in output_data
        assert 'data' in output_data
        assert 'feature_description' in output_data
        
        mock_generate.assert_called_once()
        mock_execute.assert_called_once()
    
    def test_process_core_logic_invalid_input(self):
        """Test core logic with invalid input."""
        fetcher = DataFetcher(self.config)
        
        # Invalid input - missing feature_description
        state: ProcessorState = {
            'input_data': {},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(ValueError, match="feature_description"):
            fetcher._process_core_logic(state)
    
    def test_evaluate_result_success(self):
        """Test successful result evaluation."""
        fetcher = DataFetcher(self.config)
        
        # Create mock kx.Table with test data
        mock_table = MagicMock(spec=kx.Table)
        mock_table.__getitem__.return_value = kx.q.til(100)  # Mock column data
        
        # Mock kx.q functions
        with patch('pykx.q') as mock_q:
            mock_q.count.return_value = 100
            mock_q.null.return_value = kx.q.til(100) == 0  # Some null values
            mock_q.sum.return_value = 5  # 5 null values
            mock_q.meta.return_value = kx.q.meta(kx.q.til(100))  # Mock metadata
            mock_q.unique.return_value = kx.q.til(100)  # Mock unique dates
            mock_q.asc.return_value = kx.q.til(100)  # Mock sorted dates
            mock_q.deltas.return_value = kx.q.til(99)  # Mock date differences
            mock_q.any.return_value = False  # No gaps in dates
            
            # Create state with good data
            state: ProcessorState = {
                'output_data': {
                    'query': '{"query": "select from table"}',
                    'data': {
                        'table1': mock_table
                    }
                },
                'status': 'success',
                'input_data': {},
                'metadata': {}
            }
            
            with patch.object(fetcher, '_validate_query_format', return_value=True):
                result_state = fetcher._evaluate_result(state)
            
            assert result_state['eval_passed'] is True
            assert 'eval_reason' in result_state
            assert '100 total rows' in result_state['eval_reason']
            assert 'quality_issues' in result_state.get('metadata', {})
    
    def test_evaluate_result_insufficient_data(self):
        """Test evaluation with insufficient data."""
        fetcher = DataFetcher(self.config)
        
        # Create state with insufficient data
        state: ProcessorState = {
            'output_data': {
                'query': '{"query": "select from table"}',
                'data': {
                    'table1': [{'col1': i} for i in range(10)]  # Only 10 rows
                }
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(fetcher, '_validate_query_format', return_value=True):
            result_state = fetcher._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Insufficient data' in result_state['eval_reason']
    
    def test_evaluate_result_no_data(self):
        """Test evaluation with no data."""
        fetcher = DataFetcher(self.config)
        
        state: ProcessorState = {
            'output_data': {},
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        result_state = fetcher._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert result_state['eval_reason'] == "No output data to evaluate"
    
    def test_debug_error_connection_error(self):
        """Test debugging connection errors."""
        fetcher = DataFetcher(self.config)
        
        state: ProcessorState = {
            'error': ConnectionError("Connection timeout"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = fetcher._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Connection error' in result_state['debug_reason']
    
    def test_debug_error_permission_error(self):
        """Test debugging permission errors."""
        fetcher = DataFetcher(self.config)
        
        state: ProcessorState = {
            'error': PermissionError("Access denied"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = fetcher._debug_error(state)
        
        assert result_state['should_retry'] is False
        assert 'Access permission error' in result_state['debug_reason']
    
    def test_debug_error_json_error(self):
        """Test debugging JSON parsing errors."""
        fetcher = DataFetcher(self.config)
        
        state: ProcessorState = {
            'error': ValueError("JSON parse error"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = fetcher._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Query generation error' in result_state['debug_reason']
    
    def test_handle_interrupt(self):
        """Test interrupt handling."""
        fetcher = DataFetcher(self.config)
        
        state: ProcessorState = {
            'status': 'running',
            'input_data': {},
            'output_data': {},
            'metadata': {'progress': 0.5}
        }
        
        result_state = fetcher._handle_interrupt(state)
        
        assert result_state['status'] == 'paused'
        assert 'interrupt_reason' in result_state
    
    @patch('src.processors.data_fetcher.DataFetcher._execute_kdb_query')
    def test_generate_kdb_query(self, mock_execute):
        """Test KDB query generation."""
        fetcher = DataFetcher(self.config)
        
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(
            content='{"query": "select from trades where date=2023.01.01"}'
        )
        
        # Mock prompt manager
        self.mock_prompt_manager.get_template.return_value = Mock(
            format=Mock(return_value="Generate query for: {description}")
        )
        
        query = fetcher._generate_kdb_query("Get trading data")
        
        assert query == '{"query": "select from trades where date=2023.01.01"}'
        self.mock_llm.invoke.assert_called_once()
        self.mock_prompt_manager.get_template.assert_called_once()
    
    def test_validate_query_format_valid(self):
        """Test query format validation with valid query."""
        fetcher = DataFetcher(self.config)
        
        valid_query = '{"query": "select from table", "params": {}}'
        assert fetcher._validate_query_format(valid_query) is True
    
    def test_validate_query_format_invalid(self):
        """Test query format validation with invalid query."""
        fetcher = DataFetcher(self.config)
        
        invalid_query = "not json"
        assert fetcher._validate_query_format(invalid_query) is False
    
    @patch('pykx.q')
    def test_execute_kdb_query_success(self, mock_q):
        """Test successful KDB query execution."""
        fetcher = DataFetcher(self.config)
        
        # Mock pykx response
        mock_result = Mock()
        mock_result.py.return_value = [{'col1': 1, 'col2': 2}]
        mock_q.return_value = mock_result
        
        query_json = '{"query": "select from table"}'
        result = fetcher._execute_kdb_query(query_json)
        
        assert isinstance(result, dict)
        assert 'query_result' in result
    
    def test_processor_integration(self):
        """Test full processor integration."""
        fetcher = DataFetcher(self.config)
        
        # Test that processor can be created and has all required methods
        assert hasattr(fetcher, '_process_core_logic')
        assert hasattr(fetcher, '_evaluate_result')
        assert hasattr(fetcher, '_debug_error')
        assert hasattr(fetcher, '_handle_interrupt')
        
        # Test capability validation
        capabilities = fetcher.get_capabilities()
        assert len(capabilities) == 4  # observable, evaluable, debuggable, interruptible

    def test_evaluate_result_with_quality_issues(self):
        """Test evaluation with data quality issues."""
        fetcher = DataFetcher(self.config)
        
        # Create mock kx.Table with test data
        mock_table = MagicMock(spec=kx.Table)
        mock_table.__getitem__.return_value = kx.q.til(100)  # Mock column data
        
        # Mock kx.q functions
        with patch('pykx.q') as mock_q:
            mock_q.count.return_value = 100
            mock_q.null.return_value = kx.q.til(100) == 0  # Some null values
            mock_q.sum.return_value = 10  # 10 null values
            mock_q.meta.return_value = kx.q.meta(kx.q.til(100))  # Mock metadata
            mock_q.unique.return_value = kx.q.til(100)  # Mock unique dates
            mock_q.asc.return_value = kx.q.til(100)  # Mock sorted dates
            mock_q.deltas.return_value = kx.q.til(99)  # Mock date differences
            mock_q.any.return_value = True  # Has gaps in dates
            
            # Create state with data quality issues
            state: ProcessorState = {
                'output_data': {
                    'query': '{"query": "select from table"}',
                    'data': {
                        'table1': mock_table
                    }
                },
                'status': 'success',
                'input_data': {},
                'metadata': {}
            }
            
            with patch.object(fetcher, '_validate_query_format', return_value=True):
                result_state = fetcher._evaluate_result(state)
            
            assert result_state['eval_passed'] is True  # Still passes as it has enough rows
            assert 'quality_issues' in result_state.get('metadata', {})
            quality_issues = result_state['metadata']['quality_issues']
            assert any('null values' in issue for issue in quality_issues)
            assert any('gaps in date sequence' in issue for issue in quality_issues)
    
    def test_evaluate_result_with_infinity_values(self):
        """Test evaluation with infinity values in numeric columns."""
        fetcher = DataFetcher(self.config)
        
        # Create mock kx.Table with test data
        mock_table = MagicMock(spec=kx.Table)
        mock_table.__getitem__.return_value = kx.q.til(100)  # Mock column data
        
        # Mock kx.q functions
        with patch('pykx.q') as mock_q:
            mock_q.count.return_value = 100
            mock_q.null.return_value = kx.q.til(100) == 0  # No null values
            mock_q.sum.return_value = 0  # No null values
            mock_q.meta.return_value = kx.q.meta(kx.q.til(100))  # Mock metadata
            mock_q.unique.return_value = kx.q.til(100)  # Mock unique dates
            mock_q.asc.return_value = kx.q.til(100)  # Mock sorted dates
            mock_q.deltas.return_value = kx.q.til(99)  # Mock date differences
            mock_q.any.return_value = False  # No gaps in dates
            mock_q.abs.return_value = kx.q.til(100)  # Mock absolute values
            mock_q.inf = float('inf')  # Mock infinity value
            
            # Create state with infinity values
            state: ProcessorState = {
                'output_data': {
                    'query': '{"query": "select from table"}',
                    'data': {
                        'table1': mock_table
                    }
                },
                'status': 'success',
                'input_data': {},
                'metadata': {}
            }
            
            with patch.object(fetcher, '_validate_query_format', return_value=True):
                result_state = fetcher._evaluate_result(state)
            
            assert result_state['eval_passed'] is True
            assert 'quality_issues' in result_state.get('metadata', {})
            quality_issues = result_state['metadata']['quality_issues']
            assert any('infinity values' in issue for issue in quality_issues)


class TestDataFetcherErrorHandling:
    """Test DataFetcher error handling scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'db_path': '/test/db'
        }
    
    def test_llm_failure(self):
        """Test handling of LLM failures."""
        fetcher = DataFetcher(self.config)
        
        # Mock LLM failure
        self.mock_llm.invoke.side_effect = Exception("LLM API error")
        
        state: ProcessorState = {
            'input_data': {'feature_description': 'Test feature'},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception, match="LLM API error"):
            fetcher._process_core_logic(state)


class TestDataFetcherConfiguration:
    """Test DataFetcher configuration scenarios."""
    
    def test_custom_configuration(self):
        """Test processor with custom configuration."""
        custom_config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock(),
            'db_path': '/custom/db/path',
            'min_rows_per_table': 200,
            'timeout_seconds': 30
        }
        
        fetcher = DataFetcher(custom_config)
        
        assert fetcher.db_path == '/custom/db/path'
        assert fetcher.config['min_rows_per_table'] == 200
        assert fetcher.config['timeout_seconds'] == 30
    
    def test_default_configuration(self):
        """Test processor with default configuration values."""
        minimal_config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock()
        }
        
        fetcher = DataFetcher(minimal_config)
        
        assert fetcher.db_path == 'D:/kdbdb'  # Default value 