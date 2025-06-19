"""
Tests for src/processors/backtest_runner.py

Tests BacktestRunner with new decorator-based architecture:
- Observable, Evaluable, Debuggable, Interruptible capabilities
- Strategy design and backtesting
- Performance metrics calculation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

from src.processors.backtest_runner import BacktestRunner
from src.core.base_processor import ProcessorType, ProcessorState


class TestBacktestRunner:
    """Test BacktestRunner functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        return Mock()
    
    @pytest.fixture
    def mock_backtest_engine(self):
        return Mock()
    
    @pytest.fixture
    def runner(self, mock_llm, mock_backtest_engine):
        with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
            return BacktestRunner({
                'model_name': 'gpt-4',
                'backtest_engine': mock_backtest_engine,
                'performance_metrics': ['sharpe_ratio', 'max_drawdown']
            })
    
    def test_initialization(self, runner, mock_llm, mock_backtest_engine):
        """Test BacktestRunner initialization."""
        assert runner.model_name == 'gpt-4'
        assert runner.backtest_engine == mock_backtest_engine
        assert runner.performance_metrics == ['sharpe_ratio', 'max_drawdown']
        assert runner.llm == mock_llm
    
    def test_missing_model_name(self, mock_backtest_engine):
        """Test initialization with missing model_name."""
        with pytest.raises(ValueError, match="model_name is required"):
            BacktestRunner({'backtest_engine': mock_backtest_engine})
    
    def test_missing_backtest_engine(self, mock_llm):
        """Test initialization with missing backtest_engine."""
        with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
            with pytest.raises(ValueError, match="backtest_engine is required"):
                BacktestRunner({'model_name': 'gpt-4'})
    
    def test_valid_initialization(self, mock_llm, mock_backtest_engine):
        """Test valid initialization."""
        with patch('langchain_openai.ChatOpenAI', return_value=mock_llm):
            BacktestRunner({
                'model_name': 'gpt-4',
                'backtest_engine': mock_backtest_engine
            })
    
    def test_capabilities(self):
        """Test processor capabilities."""
        runner = BacktestRunner(self.config)
        capabilities = runner.get_capabilities()
        
        assert 'observable' in capabilities
        assert 'evaluable' in capabilities
        assert 'debuggable' in capabilities
        assert 'interruptible' in capabilities
    
    @patch('src.processors.backtest_runner.BacktestRunner._calculate_performance_metrics')
    @patch('src.processors.backtest_runner.BacktestRunner._execute_backtest')
    @patch('src.processors.backtest_runner.BacktestRunner._design_trading_strategy')
    def test_process_core_logic_success(self, mock_design, mock_execute, mock_calculate):
        """Test successful core processing logic."""
        runner = BacktestRunner(self.config)
        
        # Setup mocks
        mock_design.return_value = {
            'strategy_type': 'momentum',
            'parameters': {'lookback': 20, 'threshold': 0.02}
        }
        mock_execute.return_value = {
            'trades': [{'date': '2023-01-01', 'return': 0.01}],
            'period': '2023-01-01 to 2023-12-31',
            'trades_count': 50
        }
        mock_calculate.return_value = {
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15,
            'total_return': 0.12
        }
        
        # Create input state
        state: ProcessorState = {
            'input_data': {
                'factors': {
                    'momentum_factor': {'values': [0.01, 0.02, -0.01]},
                    'value_factor': {'values': [0.005, -0.01, 0.015]}
                },
                'strategy_spec': {'type': 'long_short', 'universe': 'SP500'}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        # Process
        result_state = runner._process_core_logic(state)
        
        # Verify results
        assert result_state['status'] == 'success'
        assert 'output_data' in result_state
        
        output_data = result_state['output_data']
        assert 'strategy_design' in output_data
        assert 'backtest_results' in output_data
        assert 'performance_metrics' in output_data
        assert 'factors_used' in output_data
        
        mock_design.assert_called_once()
        mock_execute.assert_called_once()
        mock_calculate.assert_called_once()
    
    def test_process_core_logic_invalid_input(self):
        """Test core logic with invalid input."""
        runner = BacktestRunner(self.config)
        
        # Invalid input - missing factors
        state: ProcessorState = {
            'input_data': {'strategy_spec': {}},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(ValueError, match="factors"):
            runner._process_core_logic(state)
    
    def test_evaluate_result_success(self):
        """Test successful result evaluation."""
        runner = BacktestRunner(self.config)
        
        # Create state with good backtest results
        state: ProcessorState = {
            'output_data': {
                'performance_metrics': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.15,
                    'total_return': 0.12
                },
                'backtest_results': {
                    'trades_count': 25,
                    'period': '2023-01-01 to 2023-12-31'
                }
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(runner, '_validate_strategy_stability', return_value=True):
            result_state = runner._evaluate_result(state)
        
        assert result_state['eval_passed'] is True
        assert 'eval_reason' in result_state
        assert 'Sharpe=1.500' in result_state['eval_reason']
        assert 'Trades=25' in result_state['eval_reason']
    
    def test_evaluate_result_low_sharpe(self):
        """Test evaluation with low Sharpe ratio."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'performance_metrics': {
                    'sharpe_ratio': 0.5,  # Below threshold of 1.0
                    'max_drawdown': -0.15,
                    'total_return': 0.12
                },
                'backtest_results': {'trades_count': 25}
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        result_state = runner._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Low Sharpe ratio' in result_state['eval_reason']
    
    def test_evaluate_result_high_drawdown(self):
        """Test evaluation with high drawdown."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'performance_metrics': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.3,  # Below threshold of -0.2
                    'total_return': 0.12
                },
                'backtest_results': {'trades_count': 25}
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        result_state = runner._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'High drawdown' in result_state['eval_reason']
    
    def test_evaluate_result_insufficient_trades(self):
        """Test evaluation with insufficient trades."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'output_data': {
                'performance_metrics': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.15,
                    'total_return': 0.12
                },
                'backtest_results': {'trades_count': 5}  # Below threshold of 10
            },
            'status': 'success',
            'input_data': {},
            'metadata': {}
        }
        
        with patch.object(runner, '_validate_strategy_stability', return_value=True):
            result_state = runner._evaluate_result(state)
        
        assert result_state['eval_passed'] is False
        assert 'Insufficient trades' in result_state['eval_reason']
    
    def test_debug_error_strategy_error(self):
        """Test debugging strategy design errors."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'error': ValueError("Strategy design failed"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = runner._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Strategy design error' in result_state['debug_reason']
    
    def test_debug_error_backtest_engine_error(self):
        """Test debugging backtest engine errors."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'error': RuntimeError("Backtest engine failed"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = runner._debug_error(state)
        
        assert result_state['should_retry'] is True
        assert 'Backtest engine error' in result_state['debug_reason']
    
    def test_debug_error_data_error(self):
        """Test debugging data-related errors."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'error': KeyError("Factor data missing"),
            'status': 'error',
            'input_data': {},
            'output_data': {},
            'metadata': {}
        }
        
        result_state = runner._debug_error(state)
        
        assert result_state['should_retry'] is False
        assert 'Data availability error' in result_state['debug_reason']
    
    def test_handle_interrupt(self):
        """Test interrupt handling."""
        runner = BacktestRunner(self.config)
        
        state: ProcessorState = {
            'status': 'running',
            'input_data': {},
            'output_data': {},
            'metadata': {'progress': 0.6, 'current_step': 'backtesting'}
        }
        
        result_state = runner._handle_interrupt(state)
        
        assert result_state['status'] == 'paused'
        assert 'interrupt_reason' in result_state
        assert 'completion_estimate' in result_state
    
    def test_design_trading_strategy(self):
        """Test trading strategy design."""
        runner = BacktestRunner(self.config)
        
        # Mock AI response
        self.mock_ai.generate_response.return_value = Mock(
            content='{"strategy_type": "momentum", "parameters": {"lookback": 20}, "rules": ["buy when momentum > 0.02"]}'
        )
        
        # Mock prompt manager
        self.mock_prompt_manager.get_template.return_value = Mock(
            format=Mock(return_value="Design strategy for factors")
        )
        
        factors = {'momentum': [0.01, 0.02, -0.01]}
        strategy_spec = {'type': 'long_short'}
        
        strategy = runner._design_trading_strategy(factors, strategy_spec)
        
        assert isinstance(strategy, dict)
        assert 'strategy_type' in strategy
        assert 'parameters' in strategy
        assert strategy['strategy_type'] == 'momentum'
    
    def test_execute_backtest(self):
        """Test backtest execution."""
        runner = BacktestRunner(self.config)
        
        # Mock backtest engine
        self.mock_backtest_engine.run_backtest.return_value = {
            'trades': [{'date': '2023-01-01', 'return': 0.01}],
            'period': '2023-01-01 to 2023-12-31',
            'trades_count': 50
        }
        
        strategy_design = {'strategy_type': 'momentum'}
        factors = {'momentum': [0.01, 0.02]}
        
        results = runner._execute_backtest(strategy_design, factors)
        
        assert isinstance(results, dict)
        assert 'trades' in results
        assert 'period' in results
        assert 'trades_count' in results
        
        self.mock_backtest_engine.run_backtest.assert_called_once()
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        runner = BacktestRunner(self.config)
        
        backtest_results = {
            'trades': [
                {'return': 0.01, 'date': '2023-01-01'},
                {'return': 0.02, 'date': '2023-01-02'},
                {'return': -0.01, 'date': '2023-01-03'}
            ],
            'period': '2023-01-01 to 2023-01-03'
        }
        
        metrics = runner._calculate_performance_metrics(backtest_results)
        
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_return' in metrics
        assert 'volatility' in metrics
        
        # Basic sanity checks
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['total_return'], (int, float))
        assert metrics['max_drawdown'] <= 0  # Drawdown should be negative or zero
    
    def test_validate_strategy_stability(self):
        """Test strategy stability validation."""
        runner = BacktestRunner(self.config)
        
        # Stable strategy results
        stable_results = {
            'trades': [{'return': r} for r in [0.01, 0.015, 0.008, 0.012, 0.009]],
            'rolling_sharpe': [1.2, 1.3, 1.1, 1.4, 1.2]
        }
        
        assert runner._validate_strategy_stability(stable_results) is True
        
        # Unstable strategy results
        unstable_results = {
            'trades': [{'return': r} for r in [0.1, -0.08, 0.15, -0.12, 0.2]],
            'rolling_sharpe': [2.0, -1.5, 3.0, -2.0, 4.0]
        }
        
        assert runner._validate_strategy_stability(unstable_results) is False
    
    def test_processor_integration(self):
        """Test full processor integration."""
        runner = BacktestRunner(self.config)
        
        # Test that processor can be created and has all required methods
        assert hasattr(runner, '_process_core_logic')
        assert hasattr(runner, '_evaluate_result')
        assert hasattr(runner, '_debug_error')
        assert hasattr(runner, '_handle_interrupt')
        
        # Test capability validation
        capabilities = runner.get_capabilities()
        assert len(capabilities) == 4  # observable, evaluable, debuggable, interruptible

    def test_llm_failure(self):
        """Test handling of LLM failures."""
        runner = BacktestRunner(self.config)
        
        # Mock LLM failure
        self.mock_llm.invoke.side_effect = Exception("LLM API error")
        
        state: ProcessorState = {
            'input_data': {'strategy_description': 'Test strategy'},
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception, match="LLM API error"):
            runner._process_core_logic(state)


class TestBacktestRunnerPerformanceMetrics:
    """Test BacktestRunner performance metrics calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'ai_integration': Mock(),
            'prompt_manager': Mock(),
            'backtest_engine': Mock(),
            'performance_metrics': ['sharpe_ratio', 'max_drawdown', 'total_return']
        }
        self.runner = BacktestRunner(self.config)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Mock returns with known statistics
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        backtest_results = {
            'trades': [{'return': r} for r in returns],
            'period': '2023-01-01 to 2023-01-05'
        }
        
        metrics = self.runner._calculate_performance_metrics(backtest_results)
        
        # Sharpe ratio should be positive for positive mean return
        assert metrics['sharpe_ratio'] > 0
        assert isinstance(metrics['sharpe_ratio'], (int, float))
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Returns that create a known drawdown pattern
        returns = [0.1, -0.05, -0.03, 0.02, -0.08, 0.15]
        backtest_results = {
            'trades': [{'return': r} for r in returns],
            'period': '2023-01-01 to 2023-01-06'
        }
        
        metrics = self.runner._calculate_performance_metrics(backtest_results)
        
        # Max drawdown should be negative
        assert metrics['max_drawdown'] <= 0
        assert isinstance(metrics['max_drawdown'], (int, float))
    
    def test_total_return_calculation(self):
        """Test total return calculation."""
        returns = [0.1, 0.05, -0.02]  # 10%, 5%, -2%
        backtest_results = {
            'trades': [{'return': r} for r in returns],
            'period': '2023-01-01 to 2023-01-03'
        }
        
        metrics = self.runner._calculate_performance_metrics(backtest_results)
        
        # Total return should be approximately (1.1 * 1.05 * 0.98) - 1 ≈ 0.1309
        expected_return = (1.1 * 1.05 * 0.98) - 1
        assert abs(metrics['total_return'] - expected_return) < 0.001


class TestBacktestRunnerErrorHandling:
    """Test BacktestRunner error handling scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'ai_integration': Mock(),
            'prompt_manager': Mock(),
            'backtest_engine': Mock()
        }
    
    def test_strategy_design_failure(self):
        """Test handling of strategy design failures."""
        runner = BacktestRunner(self.config)
        
        # Mock AI failure
        runner.ai_integration.generate_response.side_effect = Exception("AI strategy design failed")
        
        state: ProcessorState = {
            'input_data': {
                'factors': {'momentum': [0.01, 0.02]},
                'strategy_spec': {}
            },
            'status': 'pending',
            'output_data': {},
            'metadata': {}
        }
        
        with pytest.raises(Exception):
            runner._process_core_logic(state)
    
    def test_backtest_engine_failure(self):
        """Test handling of backtest engine failures."""
        runner = BacktestRunner(self.config)
        
        with patch('src.processors.backtest_runner.BacktestRunner._design_trading_strategy', return_value={}):
            runner.backtest_engine.run_backtest.side_effect = RuntimeError("Backtest engine crashed")
            
            state: ProcessorState = {
                'input_data': {
                    'factors': {'momentum': [0.01, 0.02]},
                    'strategy_spec': {}
                },
                'status': 'pending',
                'output_data': {},
                'metadata': {}
            }
            
            with pytest.raises(RuntimeError):
                runner._process_core_logic(state)


class TestBacktestRunnerConfiguration:
    """Test BacktestRunner configuration scenarios."""
    
    def test_custom_performance_metrics(self):
        """Test processor with custom performance metrics."""
        custom_config = {
            'ai_integration': Mock(),
            'prompt_manager': Mock(),
            'backtest_engine': Mock(),
            'performance_metrics': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
            'min_sharpe_ratio': 1.5,
            'max_drawdown_threshold': -0.15
        }
        
        runner = BacktestRunner(custom_config)
        
        assert runner.performance_metrics == ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        assert runner.config['min_sharpe_ratio'] == 1.5
        assert runner.config['max_drawdown_threshold'] == -0.15
    
    def test_default_performance_metrics(self):
        """Test processor with default performance metrics."""
        minimal_config = {
            'ai_integration': Mock(),
            'prompt_manager': Mock(),
            'backtest_engine': Mock()
        }
        
        runner = BacktestRunner(minimal_config)
        
        assert runner.performance_metrics == ['sharpe_ratio', 'max_drawdown', 'total_return'] 