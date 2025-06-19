"""
Tests for src/processors/feature_builder.py

Tests FeatureBuilder with new decorator-based architecture:
- Observable, Evaluable, Debuggable, Interruptible capabilities
- Core feature building logic
- Feature generation and validation
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.processors.feature_builder import FeatureBuilder
from src.core.base_processor import ProcessorType, ProcessorState


class TestFeatureBuilder:
    """Test FeatureBuilder functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt_manager = Mock()
        
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': self.mock_prompt_manager,
            'feature_template': 'test_template'
        }
    
    def test_initialization(self):
        """Test FeatureBuilder initialization."""
        builder = FeatureBuilder(self.config)
        
        assert builder.get_processor_type() == ProcessorType.FEATURE_BUILDER
        assert builder.model_name == 'gpt-4'
        assert builder.prompt_manager == self.mock_prompt_manager
        assert builder.feature_template == 'test_template'
    
    def test_initialization_missing_dependencies(self):
        """Test initialization with missing dependencies."""
        # Missing model_name
        with pytest.raises(ValueError, match="model_name is required"):
            FeatureBuilder({'prompt_manager': self.mock_prompt_manager})
        
        # Missing prompt_manager
        with pytest.raises(ValueError, match="prompt_manager is required"):
            FeatureBuilder({'model_name': 'gpt-4'})
    
    def test_capabilities(self):
        """Test processor capabilities."""
        builder = FeatureBuilder(self.config)
        capabilities = builder.get_capabilities()
        
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' not in capabilities  # FeatureBuilder is not interruptible
    
    @patch('src.processors.feature_builder.FeatureBuilder._execute_q_code')
    @patch('src.processors.feature_builder.FeatureBuilder._generate_feature_code')
    def test_process_core_logic_success(self, mock_generate, mock_execute):
        """Test successful core processing logic."""
        builder = FeatureBuilder(self.config)
        
        # Setup mocks
        mock_generate.return_value = 'feature1: avg price; feature2: max volume'
        mock_execute.return_value = {
            'feature1': {'values': [1.0, 2.0, 3.0]},
            'feature2': {'values': [100, 200, 300]}
        }
        
        # Create input state
        state: ProcessorState = {
            'input_data': {
                'feature_spec': {'type': 'price_features', 'symbols': ['AAPL']},
                'data_tables': {'trades': [{'price': 100, 'volume': 1000}]}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        # Process
        result_state = builder._process_core_logic(state)
        
        # Verify results
        assert result_state['status'] == 'success'
        assert 'output_data' in result_state
        
        output_data = result_state['output_data']
        assert 'q_code' in output_data
        assert 'features' in output_data
        assert 'feature_spec' in output_data
        
        mock_generate.assert_called_once()
        mock_execute.assert_called_once()
    
    def test_process_core_logic_invalid_input(self):
        """Test core logic with invalid input."""
        builder = FeatureBuilder(self.config)
        
        # Invalid input - missing feature_spec
        state: ProcessorState = {
            'input_data': {'data_tables': {}},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(ValueError, match="feature_spec"):
            builder._process_core_logic(state)
    
    def test_evaluate_result_success(self):
        """Test successful result evaluation."""
        builder = FeatureBuilder(self.config)
        
        # Create state with good features
        state: ProcessorState = {
            'output_data': {
                'q_code': 'feature1: avg price; feature2: max volume; feature3: sum qty',
                'features': {
                    'feature1': {'values': [1.0, 2.0, 3.0]},
                    'feature2': {'values': [100, 200, 300]},
                    'feature3': {'values': [10, 20, 30]}
                }
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(builder, '_validate_q_code_syntax', return_value=True):
            with patch.object(builder, '_validate_feature_data', return_value=True):
                result_state = builder._evaluate_result(state)
        
        assert result_state['eval_passed'] is True
        assert 'eval_reason' in result_state
        assert '3 quality features' in result_state['eval_reason']
    
    def test_evaluate_result_insufficient_features(self):
        """Test evaluation with insufficient features."""
        builder = FeatureBuilder(self.config)
        
        # Create state with too few features
        state: ProcessorState = {
            'output_data': {
                'q_code': 'feature1: avg price',
                'features': {
                    'feature1': {'values': [1.0, 2.0, 3.0]}
                }
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(builder, '_validate_q_code_syntax', return_value=True):
            result_state = builder._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Too few features' in result_state['eval_reason']
    
    def test_evaluate_result_invalid_q_code(self):
        """Test evaluation with invalid Q code."""
        builder = FeatureBuilder(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'q_code': 'invalid q syntax',
                'features': {'feature1': {'values': [1, 2, 3]}}
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(builder, '_validate_q_code_syntax', return_value=False):
            result_state = builder._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Invalid q code syntax' in result_state['eval_reason']
    
    def test_debug_error_syntax_error(self):
        """Test debugging Q syntax errors."""
        builder = FeatureBuilder(self.config)
        
        state: ProcessorState = {
            'error': SyntaxError("Q syntax error in feature calculation"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = builder._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Q code syntax error' in result_state['debug_reason']
    
    def test_debug_error_function_error(self):
        """Test debugging undefined function errors."""
        builder = FeatureBuilder(self.config)
        
        state: ProcessorState = {
            'error': NameError("Undefined function 'customfunc'"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = builder._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Undefined function error' in result_state['debug_reason']
    
    def test_debug_error_data_error(self):
        """Test debugging data-related errors."""
        builder = FeatureBuilder(self.config)
        
        state: ProcessorState = {
            'error': KeyError("Column 'price' not found"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = builder._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Data structure mismatch' in result_state['debug_reason']
    
    def test_generate_feature_code(self):
        """Test Q code generation for features."""
        builder = FeatureBuilder(self.config)
        
        # Mock AI response
        self.mock_prompt_manager.get_template.return_value = Mock(
            format=Mock(return_value="Generate features for: {spec}")
        )
        
        feature_spec = {'type': 'price_features'}
        data_tables = {'trades': []}
        
        q_code = builder._generate_feature_code(feature_spec, data_tables)
        
        assert q_code == 'feature1: avg price; feature2: max volume'
        self.mock_prompt_manager.get_template.assert_called_once()
    
    def test_validate_q_code_syntax_valid(self):
        """Test Q code syntax validation with valid code."""
        builder = FeatureBuilder(self.config)
        
        valid_code = 'feature1: avg price; feature2: max volume'
        assert builder._validate_q_code_syntax(valid_code) is True
    
    def test_validate_q_code_syntax_invalid(self):
        """Test Q code syntax validation with invalid code."""
        builder = FeatureBuilder(self.config)
        
        invalid_code = 'invalid syntax ;;; error'
        assert builder._validate_q_code_syntax(invalid_code) is False
    
    def test_validate_feature_data_valid(self):
        """Test feature data validation with valid data."""
        builder = FeatureBuilder(self.config)
        
        valid_data = {'values': [1.0, 2.0, 3.0], 'metadata': {'type': 'numeric'}}
        assert builder._validate_feature_data('test_feature', valid_data) is True
    
    def test_validate_feature_data_invalid(self):
        """Test feature data validation with invalid data."""
        builder = FeatureBuilder(self.config)
        
        invalid_data = {'values': []}  # Empty values
        assert builder._validate_feature_data('test_feature', invalid_data) is False
    
    @patch('pykx.q')
    def test_execute_q_code_success(self, mock_q):
        """Test successful Q code execution."""
        builder = FeatureBuilder(self.config)
        
        # Mock pykx response
        mock_result = Mock()
        mock_result.py.return_value = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        mock_q.return_value = mock_result
        
        q_code = 'feature1: avg price; feature2: max volume'
        data_tables = {'trades': []}
        
        result = builder._execute_q_code(q_code, data_tables)
        
        assert isinstance(result, dict)
        assert 'feature1' in result
        assert 'feature2' in result
    
    def test_processor_integration(self):
        """Test full processor integration."""
        builder = FeatureBuilder(self.config)
        
        # Test that processor can be created and has all required methods
        assert hasattr(builder, '_process_core_logic')
        assert hasattr(builder, '_evaluate_result')
        assert hasattr(builder, '_debug_error')
        
        # Test capability validation
        capabilities = builder.get_capabilities()
        assert len(capabilities) == 3  # observable, evaluable, debuggable
    
    def test_llm_failure(self):
        """Test handling of LLM failures."""
        builder = FeatureBuilder(self.config)
        
        # Mock LLM failure
        self.mock_llm.invoke.side_effect = Exception("LLM API error")
        
        state: ProcessorState = {
            'input_data': {'data': {'col1': [1, 2, 3]}},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception, match="LLM API error"):
            builder._process_core_logic(state)


class TestFeatureBuilderErrorHandling:
    """Test FeatureBuilder error handling scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock(),
            'test_data': {}
        }
    
    def test_q_code_execution_failure(self):
        """Test handling of Q code execution failures."""
        builder = FeatureBuilder(self.config)
        
        with patch('src.processors.feature_builder.FeatureBuilder._generate_feature_code', return_value='valid q code'):
            with patch('src.processors.feature_builder.FeatureBuilder._execute_q_code', side_effect=RuntimeError("Q execution failed")):
                state: ProcessorState = {
                    'input_data': {
                        'feature_spec': {'type': 'test'},
                        'data_tables': {}
                    },
                    'status': 'pending',
                    'output_data': {},
                    'metadata': {}
                }
                
                with pytest.raises(RuntimeError):
                    builder._process_core_logic(state)
    
    def test_ai_generation_failure(self):
        """Test handling of AI code generation failures."""
        builder = FeatureBuilder(self.config)
        
        # Mock AI failure
        builder.prompt_manager.get_template.side_effect = Exception("AI service error")
        
        state: ProcessorState = {
            'input_data': {
                'feature_spec': {'type': 'test'},
                'data_tables': {}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception):
            builder._process_core_logic(state)


class TestFeatureBuilderConfiguration:
    """Test FeatureBuilder configuration scenarios."""
    
    def test_custom_configuration(self):
        """Test processor with custom configuration."""
        custom_config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock(),
            'test_data': {'custom_table': []},
            'min_features_count': 5,
            'max_retries': 3
        }
        
        builder = FeatureBuilder(custom_config)
        
        assert builder.test_data == custom_config['test_data']
        assert builder.config['min_features_count'] == 5
        assert builder.config['max_retries'] == 3
    
    def test_default_configuration(self):
        """Test processor with default configuration values."""
        minimal_config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock()
        }
        
        builder = FeatureBuilder(minimal_config)
        
        assert builder.test_data == {}  # Default value 