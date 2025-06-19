"""
Tests for src/processors/factor_augmenter.py

Tests FactorAugmenter with new decorator-based architecture:
- Observable, Evaluable, Debuggable, Interruptible capabilities
- Core factor augmentation logic
- Factor enhancement and validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

from src.processors.factor_augmenter import FactorAugmenter
from src.core.base_processor import ProcessorType, ProcessorState


class TestFactorAugmenter:
    """Test FactorAugmenter functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        return Mock()
    
    @pytest.fixture
    def augmenter(self, mock_llm):
        with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
            return FactorAugmenter({
                'model_name': 'gpt-4',
                'enhancement_methods': ['transform', 'combine']
            })
    
    def test_initialization(self, augmenter, mock_llm):
        """Test FactorAugmenter initialization."""
        assert augmenter.model_name == 'gpt-4'
        assert augmenter.enhancement_methods == ['transform', 'combine']
        assert augmenter.llm == mock_llm
    
    def test_missing_model_name(self):
        """Test initialization with missing model_name."""
        with pytest.raises(ValueError, match="model_name is required"):
            FactorAugmenter({})
    
    def test_valid_initialization(self, mock_llm):
        """Test valid initialization."""
        with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
            FactorAugmenter({'model_name': 'gpt-4'})
    
    def test_initialization_missing_dependencies(self):
        """Test initialization with missing dependencies."""
        # Missing model_name
        with pytest.raises(ValueError, match="model_name is required"):
            FactorAugmenter({'prompt_manager': self.mock_prompt_manager})
        
        # Missing prompt_manager
        with pytest.raises(ValueError, match="prompt_manager is required"):
            FactorAugmenter({'model_name': 'gpt-4'})
    
    def test_capabilities(self):
        """Test processor capabilities."""
        augmenter = FactorAugmenter(self.config)
        capabilities = augmenter.get_capabilities()
        
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' not in capabilities  # FactorAugmenter is not interruptible
    
    @patch('src.processors.factor_augmenter.FactorAugmenter._apply_enhancements')
    @patch('src.processors.factor_augmenter.FactorAugmenter._generate_enhancement_strategies')
    def test_process_core_logic_success(self, mock_generate, mock_apply):
        """Test successful core processing logic."""
        augmenter = FactorAugmenter(self.config)
        
        # Setup mocks
        mock_generate.return_value = [
            {'method': 'transform', 'params': {'type': 'log'}},
            {'method': 'combine', 'params': {'operation': 'ratio'}}
        ]
        mock_apply.return_value = {
            'enhanced_factor1': {'values': [1.0, 2.0, 3.0]},
            'enhanced_factor2': {'values': [0.5, 1.0, 1.5]},
            'combined_factor1': {'values': [2.0, 4.0, 6.0]}
        }
        
        # Create input state
        state: ProcessorState = {
            'input_data': {
                'raw_features': {
                    'feature1': {'values': [10, 20, 30]},
                    'feature2': {'values': [5, 10, 15]}
                },
                'enhancement_spec': {'target_count': 5}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        # Process
        result_state = augmenter._process_core_logic(state)
        
        # Verify results
        assert result_state['status'] == 'success'
        assert 'output_data' in result_state
        
        output_data = result_state['output_data']
        assert 'enhanced_factors' in output_data
        assert 'enhancement_strategies' in output_data
        assert 'raw_features_count' in output_data
        assert 'enhanced_factors_count' in output_data
        
        mock_generate.assert_called_once()
        mock_apply.assert_called_once()
    
    def test_process_core_logic_invalid_input(self):
        """Test core logic with invalid input."""
        augmenter = FactorAugmenter(self.config)
        
        # Invalid input - missing raw_features
        state: ProcessorState = {
            'input_data': {'enhancement_spec': {}},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(ValueError, match="raw_features"):
            augmenter._process_core_logic(state)
    
    def test_evaluate_result_success(self):
        """Test successful result evaluation."""
        augmenter = FactorAugmenter(self.config)
        
        # Create state with good enhancement results
        state: ProcessorState = {
            'output_data': {
                'enhanced_factors': {
                    'factor1': {'values': [1, 2, 3]},
                    'factor2': {'values': [4, 5, 6]},
                    'factor3': {'values': [7, 8, 9]},
                    'factor4': {'values': [10, 11, 12]}
                },
                'raw_features_count': 2,
                'enhancement_strategies': [
                    {'method': 'transform'},
                    {'method': 'combine'},
                    {'method': 'normalize'}
                ]
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(augmenter, '_validate_factor_quality', return_value=True):
            result_state = augmenter._evaluate_result(state)
        
        assert result_state['eval_passed'] is True
        assert 'eval_reason' in result_state
        assert '4 factors' in result_state['eval_reason']
        assert '2.00x improvement' in result_state['eval_reason']
    
    def test_evaluate_result_insufficient_improvement(self):
        """Test evaluation with insufficient improvement ratio."""
        augmenter = FactorAugmenter(self.config)
        
        # Create state with insufficient improvement
        state: ProcessorState = {
            'output_data': {
                'enhanced_factors': {
                    'factor1': {'values': [1, 2, 3]}
                },
                'raw_features_count': 2,  # Only 0.5x improvement
                'enhancement_strategies': [{'method': 'transform'}]
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        result_state = augmenter._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Insufficient enhancement' in result_state['eval_reason']
    
    def test_evaluate_result_quality_issues(self):
        """Test evaluation with factor quality issues."""
        augmenter = FactorAugmenter(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'enhanced_factors': {
                    'factor1': {'values': [1, 2, 3]},
                    'factor2': {'values': [4, 5, 6]},
                    'factor3': {'values': [7, 8, 9]}
                },
                'raw_features_count': 1,
                'enhancement_strategies': [{'method': 'transform'}]
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        # Mock quality validation to fail for 2 factors (exceeds max_quality_issues=1)
        def mock_validate(name, data):
            return name != 'factor2' and name != 'factor3'
        
        with patch.object(augmenter, '_validate_factor_quality', side_effect=mock_validate):
            result_state = augmenter._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Too many quality issues' in result_state['eval_reason']
    
    def test_evaluate_result_insufficient_diversity(self):
        """Test evaluation with insufficient enhancement method diversity."""
        augmenter = FactorAugmenter(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'enhanced_factors': {
                    'factor1': {'values': [1, 2, 3]},
                    'factor2': {'values': [4, 5, 6]}
                },
                'raw_features_count': 1,
                'enhancement_strategies': [
                    {'method': 'transform'},
                    {'method': 'transform'}  # Only one method type
                ]
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(augmenter, '_validate_factor_quality', return_value=True):
            result_state = augmenter._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Insufficient enhancement diversity' in result_state['eval_reason']
    
    def test_debug_error_computation_error(self):
        """Test debugging computation errors."""
        augmenter = FactorAugmenter(self.config)
        
        state: ProcessorState = {
            'error': ArithmeticError("Factor computation failed"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = augmenter._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Factor computation error' in result_state['debug_reason']
    
    def test_debug_error_memory_error(self):
        """Test debugging memory errors."""
        augmenter = FactorAugmenter(self.config)
        
        state: ProcessorState = {
            'error': MemoryError("Out of memory during enhancement"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = augmenter._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Memory error' in result_state['debug_reason']
    
    def test_debug_error_data_error(self):
        """Test debugging data-related errors."""
        augmenter = FactorAugmenter(self.config)
        
        state: ProcessorState = {
            'error': ValueError("Invalid data format for enhancement"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = augmenter._debug_error(state)
        
        assert result_state['should_retry'] is False
        assert 'Data format error' in result_state['debug_reason']
    
    def test_generate_enhancement_strategies(self):
        """Test enhancement strategy generation."""
        augmenter = FactorAugmenter(self.config)
        
        # Mock AI response
        self.mock_prompt_manager.get_template.return_value = Mock(
            format=Mock(return_value="Generate enhancement strategies")
        )
        
        raw_features = {'feature1': [1, 2, 3]}
        enhancement_spec = {'target_count': 5}
        
        strategies = augmenter._generate_enhancement_strategies(raw_features, enhancement_spec)
        
        assert isinstance(strategies, list)
        assert len(strategies) == 2
        assert strategies[0]['method'] == 'transform'
        assert strategies[1]['method'] == 'combine'
    
    def test_apply_enhancements(self):
        """Test enhancement application."""
        augmenter = FactorAugmenter(self.config)
        
        raw_features = {
            'feature1': {'values': [1, 2, 3]},
            'feature2': {'values': [4, 5, 6]}
        }
        
        strategies = [
            {'method': 'transform', 'target_features': ['feature1'], 'params': {'type': 'log'}},
            {'method': 'combine', 'target_features': ['feature1', 'feature2'], 'params': {'operation': 'ratio'}}
        ]
        
        with patch.object(augmenter, '_apply_transformation', return_value={'log_feature1': {'values': [0, 0.69, 1.1]}}):
            with patch.object(augmenter, '_apply_combination', return_value={'ratio_feature1_feature2': {'values': [0.25, 0.4, 0.5]}}):
                enhanced = augmenter._apply_enhancements(raw_features, strategies)
        
        assert isinstance(enhanced, dict)
        assert len(enhanced) >= 2  # Original features plus enhanced ones
    
    def test_validate_factor_quality_valid(self):
        """Test factor quality validation with valid data."""
        augmenter = FactorAugmenter(self.config)
        
        valid_data = {
            'values': [1.0, 2.0, 3.0, 4.0, 5.0],
            'metadata': {'type': 'numeric', 'source': 'enhanced'}
        }
        
        assert augmenter._validate_factor_quality('test_factor', valid_data) is True
    
    def test_validate_factor_quality_invalid(self):
        """Test factor quality validation with invalid data."""
        augmenter = FactorAugmenter(self.config)
        
        # Test with empty values
        invalid_data = {'values': []}
        assert augmenter._validate_factor_quality('test_factor', invalid_data) is False
        
        # Test with all NaN values
        invalid_data = {'values': [float('nan'), float('nan')]}
        assert augmenter._validate_factor_quality('test_factor', invalid_data) is False
    
    def test_processor_integration(self):
        """Test full processor integration."""
        augmenter = FactorAugmenter(self.config)
        
        # Test that processor can be created and has all required methods
        assert hasattr(augmenter, '_process_core_logic')
        assert hasattr(augmenter, '_evaluate_result')
        assert hasattr(augmenter, '_debug_error')
        
        # Test capability validation
        capabilities = augmenter.get_capabilities()
        assert len(capabilities) == 3  # observable, evaluable, debuggable

    def test_llm_failure(self):
        """Test handling of LLM failures."""
        augmenter = FactorAugmenter(self.config)
        
        # Mock LLM failure
        self.mock_llm.invoke.side_effect = Exception("LLM API error")
        
        state: ProcessorState = {
            'input_data': {'factors': {'factor1': [1, 2, 3]}},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception, match="LLM API error"):
            augmenter._process_core_logic(state)


class TestFactorAugmenterEnhancementMethods:
    """Test specific enhancement methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock(),
            'enhancement_methods': ['transform', 'combine', 'normalize']
        }
        self.augmenter = FactorAugmenter(self.config)
    
    def test_apply_transformation(self):
        """Test transformation enhancement method."""
        features = {
            'feature1': {'values': [1, 4, 9, 16]},
            'feature2': {'values': [2, 8, 18, 32]}
        }
        
        target_features = ['feature1']
        params = {'type': 'sqrt'}
        
        result = self.augmenter._apply_transformation(features, target_features, params)
        
        assert isinstance(result, dict)
        assert 'sqrt_feature1' in result
        # sqrt([1, 4, 9, 16]) = [1, 2, 3, 4]
        assert result['sqrt_feature1']['values'] == [1.0, 2.0, 3.0, 4.0]
    
    def test_apply_combination(self):
        """Test combination enhancement method."""
        features = {
            'feature1': {'values': [2, 4, 6, 8]},
            'feature2': {'values': [1, 2, 3, 4]}
        }
        
        target_features = ['feature1', 'feature2']
        params = {'operation': 'ratio'}
        
        result = self.augmenter._apply_combination(features, target_features, params)
        
        assert isinstance(result, dict)
        assert 'ratio_feature1_feature2' in result
        # [2,4,6,8] / [1,2,3,4] = [2,2,2,2]
        assert result['ratio_feature1_feature2']['values'] == [2.0, 2.0, 2.0, 2.0]
    
    def test_apply_normalization(self):
        """Test normalization enhancement method."""
        features = {
            'feature1': {'values': [1, 2, 3, 4, 5]},
            'feature2': {'values': [10, 20, 30, 40, 50]}
        }
        
        target_features = ['feature1']
        params = {'method': 'zscore'}
        
        result = self.augmenter._apply_normalization(features, target_features, params)
        
        assert isinstance(result, dict)
        assert 'zscore_feature1' in result
        
        # Z-score normalization should have mean ~0 and std ~1
        normalized_values = result['zscore_feature1']['values']
        mean = sum(normalized_values) / len(normalized_values)
        assert abs(mean) < 0.001  # Mean should be close to 0


class TestFactorAugmenterErrorHandling:
    """Test FactorAugmenter error handling scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'model_name': 'gpt-4',
            'prompt_manager': Mock(),
            'enhancement_methods': ['transform']
        }
    
    def test_enhancement_strategy_generation_failure(self):
        """Test handling of strategy generation failures."""
        augmenter = FactorAugmenter(self.config)
        
        # Mock AI failure
        augmenter.llm.invoke.side_effect = Exception("AI service error")
        
        state: ProcessorState = {
            'input_data': {
                'raw_features': {'feature1': [1, 2, 3]},
                'enhancement_spec': {}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception):
            augmenter._process_core_logic(state)
    
    def test_enhancement_application_failure(self):
        """Test handling of enhancement application failures."""
        augmenter = FactorAugmenter(self.config)
        
        with patch('src.processors.factor_augmenter.FactorAugmenter._generate_enhancement_strategies', return_value=[]):
            with patch('src.processors.factor_augmenter.FactorAugmenter._apply_enhancements', side_effect=RuntimeError("Enhancement failed")):
                state: ProcessorState = {
                    'input_data': {
                        'raw_features': {'feature1': [1, 2, 3]},
                        'enhancement_spec': {}
                    },
                    'status': 'pending',
                    'output_data': {},
                    'metadata': {}
                }
                
                with pytest.raises(RuntimeError):
                    augmenter._process_core_logic(state) 