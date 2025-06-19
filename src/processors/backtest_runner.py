"""
BacktestRunner Processor - AI-driven strategy design and backtesting.
"""

from typing import Dict, Any, List
import logging
import json
import datetime

from ..core.base_processor import BaseProcessor, ProcessorType, ProcessorState
from ..core.decorators import evaluable, debuggable, interruptible, observable
from ..prompt_lib import get_prompt_manager
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@observable(observers=["ui", "logger", "performance_monitor"])
@evaluable(max_retries=2)
@debuggable(max_retries=1)
@interruptible(save_point_id="backtest_run")
class BacktestRunner(BaseProcessor):
    """
    AI-driven backtest runner that designs strategies and evaluates performance.
    
    Capabilities:
    - Observable: UI monitoring, logging, and performance tracking
    - Evaluable: Strategy performance validation and metrics assessment
    - Debuggable: Backtest error analysis and strategy optimization
    - Interruptible: User can pause/resume long-running backtests
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BacktestRunner.
        
        Required config:
        - model_name: Name of the OpenAI model to use (default: 'gpt-4')
        - backtest_engine: Backtesting engine implementation
        
        Optional config:
        - performance_metrics: List of metrics to evaluate
        """
        super().__init__(config)
        
        # Extract dependencies from config
        self.model_name = config.get('model_name', 'gpt-4')
        self.prompt_manager = config.get('prompt_manager', get_prompt_manager())
        self.backtest_engine = config.get('backtest_engine')
        self.performance_metrics = config.get('performance_metrics', ['sharpe_ratio', 'max_drawdown', 'total_return'])
        
        if not self.backtest_engine:
            raise ValueError("backtest_engine is required in config")
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.model_name)
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    def get_processor_type(self) -> ProcessorType:
        """Return processor type."""
        return ProcessorType.BACKTEST_RUNNER
    
    def _process_core_logic(self, state: ProcessorState) -> ProcessorState:
        """
        Core backtesting logic: design strategy and run backtest.
        
        Args:
            state: Input contains factors and strategy_spec
            
        Returns:
            Updated state with backtest results and performance metrics
        """
        try:
            # Extract input data
            input_data = state['input_data']
            if not isinstance(input_data, dict):
                raise ValueError("Input must be a dictionary")
            
            factors = input_data.get('factors', {})
            strategy_spec = input_data.get('strategy_spec', {})
            
            if not factors:
                raise ValueError("Input must contain 'factors'")
            
            self.logger.info(f"Running backtest with {len(factors)} factors...")
            
            # Step 1: Design trading strategy using AI
            strategy_design = self._design_trading_strategy(factors, strategy_spec)
            self.logger.info("Successfully designed trading strategy")
            
            # Step 2: Execute backtest
            backtest_results = self._execute_backtest(strategy_design, factors)
            self.logger.info("Successfully completed backtest execution")
            
            # Step 3: Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results)
            self.logger.info(f"Calculated {len(performance_metrics)} performance metrics")
            
            # Update state with results
            state['output_data'] = {
                'strategy_design': strategy_design,
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics,
                'factors_used': list(factors.keys()),
                'backtest_period': backtest_results.get('period', 'unknown')
            }
            state['status'] = 'success'
            
            return state
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            state['error'] = e
            state['status'] = 'error'
            raise
    
    def _evaluate_result(self, state: ProcessorState) -> ProcessorState:
        """
        Evaluate backtest results for strategy quality and performance.
        
        Args:
            state: State with backtest results
            
        Returns:
            Updated state with evaluation results
        """
        try:
            output_data = state.get('output_data')
            if not output_data:
                state['eval_passed'] = False
                state['eval_reason'] = "No output data to evaluate"
                return state
            
            performance_metrics = output_data.get('performance_metrics', {})
            backtest_results = output_data.get('backtest_results', {})
            
            # Check 1: Essential metrics exist
            required_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return']
            missing_metrics = [m for m in required_metrics if m not in performance_metrics]
            
            if missing_metrics:
                state['eval_passed'] = False
                state['eval_reason'] = f"Missing required metrics: {', '.join(missing_metrics)}"
                return state
            
            # Check 2: Performance thresholds
            min_sharpe = self.config.get('min_sharpe_ratio', 1.0)
            max_drawdown = self.config.get('max_drawdown_threshold', -0.2)
            min_return = self.config.get('min_total_return', 0.05)
            
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            drawdown = performance_metrics.get('max_drawdown', 0)
            total_return = performance_metrics.get('total_return', 0)
            
            performance_issues = []
            
            if sharpe_ratio < min_sharpe:
                performance_issues.append(f"Low Sharpe ratio: {sharpe_ratio:.3f} < {min_sharpe}")
            
            if drawdown < max_drawdown:
                performance_issues.append(f"High drawdown: {drawdown:.3f} < {max_drawdown}")
            
            if total_return < min_return:
                performance_issues.append(f"Low return: {total_return:.3f} < {min_return}")
            
            if performance_issues:
                state['eval_passed'] = False
                state['eval_reason'] = f"Performance issues: {'; '.join(performance_issues)}"
                return state
            
            # Check 3: Backtest data quality
            trades_count = backtest_results.get('trades_count', 0)
            min_trades = self.config.get('min_trades_count', 10)
            
            if trades_count < min_trades:
                state['eval_passed'] = False
                state['eval_reason'] = f"Insufficient trades: {trades_count} < {min_trades} required"
                return state
            
            # Check 4: Strategy stability
            if not self._validate_strategy_stability(backtest_results):
                state['eval_passed'] = False
                state['eval_reason'] = "Strategy shows signs of instability"
                return state
            
            # All checks passed
            state['eval_passed'] = True
            state['eval_reason'] = (
                f"Backtest evaluation passed: Sharpe={sharpe_ratio:.3f}, "
                f"Drawdown={drawdown:.3f}, Return={total_return:.3f}, "
                f"Trades={trades_count}"
            )
            
            self.logger.info(f"Backtest evaluation passed: {state['eval_reason']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Backtest evaluation failed: {e}")
            state['eval_passed'] = False
            state['eval_reason'] = f"Evaluation error: {str(e)}"
            return state
    
    def _debug_error(self, state: ProcessorState) -> ProcessorState:
        """
        Debug backtest errors and determine retry strategy.
        
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
        if 'data' in error_str or 'missing' in error_str:
            # Data issues - may resolve with data validation
            state['should_retry'] = True
            state['debug_reason'] = "Data issue - will retry with data validation"
            self.logger.info("Debug: Data error - scheduling retry with validation")
            
        elif 'strategy' in error_str or 'signal' in error_str:
            # Strategy design issues - worth retrying with simpler strategy
            state['should_retry'] = True
            state['debug_reason'] = "Strategy error - will retry with simplified strategy"
            self.logger.info("Debug: Strategy error - scheduling retry with simpler approach")
            
        elif 'performance' in error_str or 'metric' in error_str:
            # Performance calculation errors - may be temporary
            state['should_retry'] = True
            state['debug_reason'] = "Performance calculation error - will retry"
            self.logger.info("Debug: Performance error - scheduling retry")
            
        elif 'memory' in error_str or 'resource' in error_str:
            # Resource errors - unlikely to resolve with retry
            state['should_retry'] = False
            state['debug_reason'] = "Resource limitation - manual optimization required"
            self.logger.warning("Debug: Resource error - manual intervention needed")
            
        elif 'timeout' in error_str:
            # Timeout errors - may resolve with shorter backtest period
            state['should_retry'] = True
            state['debug_reason'] = "Timeout - will retry with shorter period"
            self.logger.info("Debug: Timeout - scheduling retry with reduced scope")
            
        else:
            # Unknown error - try once more
            state['should_retry'] = True
            state['debug_reason'] = f"Unknown error: {error_str[:100]} - will retry with conservative settings"
            self.logger.warning(f"Debug: Unknown error - {error_str[:100]}")
        
        return state
    
    def _handle_interrupt(self, state: ProcessorState) -> ProcessorState:
        """
        Handle user interrupt requests during backtesting.
        
        Args:
            state: State with interrupt_requested=True
            
        Returns:
            Updated state with interrupt handling
        """
        self.logger.info("Handling user interrupt request during backtest")
        
        # Save current backtest progress
        current_progress = {
            'stage': state.get('status', 'unknown'),
            'partial_results': state.get('output_data'),
            'timestamp': datetime.datetime.now().isoformat(),
            'completion_percentage': self._estimate_completion_percentage(state)
        }
        
        state['interrupt_data'] = current_progress
        state['status'] = 'paused_by_user'
        
        # If backtest engine supports pause, use it
        if hasattr(self.backtest_engine, 'pause'):
            try:
                self.backtest_engine.pause()
                self.logger.info("Backtest engine paused successfully")
            except Exception as e:
                self.logger.warning(f"Failed to pause backtest engine: {e}")
        
        self.logger.info(f"Backtest paused at {current_progress['completion_percentage']:.1f}% completion")
        
        return state
    
    def _design_trading_strategy(self, factors: Dict[str, Any], strategy_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design trading strategy using AI.
        
        Args:
            factors: Available factors for strategy design
            strategy_spec: Strategy requirements and preferences
            
        Returns:
            Strategy design specification
        """
        # Prepare factor summary for AI
        factor_summary = self._summarize_factors(factors)
        
        # Get prompt template
        prompt_content = self.prompt_manager.format_template(
            processor_type='backtest_runner',
            template_name='strategy_design',
            factors=json.dumps(factor_summary, indent=2),
            strategy_spec=json.dumps(strategy_spec, indent=2)
        )
        
        if prompt_content is None:
            # Fallback prompt if template not found
            prompt_content = f"""Design a trading strategy using the following factors:

Factors: {json.dumps(factor_summary, indent=2)}
Strategy Spec: {json.dumps(strategy_spec, indent=2)}

Return a complete strategy design in JSON format."""
            self.logger.warning("Using fallback prompt - strategy_design template not found")
        
        # Call LLM
        response = self.llm.invoke(prompt_content)
        response_content = response.content
        
        # Parse strategy design from response
        strategy_design = self._parse_strategy_design(response_content)
        if not strategy_design:
            raise ValueError("No valid strategy design found in AI response")
        
        return strategy_design
    
    def _execute_backtest(self, strategy_design: Dict[str, Any], factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute backtest using the designed strategy.
        
        Args:
            strategy_design: Trading strategy specification
            factors: Factor data for backtesting
            
        Returns:
            Dictionary containing backtest results
        """
        # Mock backtest execution (replace with actual backtesting engine)
        self.logger.info("Executing backtest simulation")
        
        # Simulate backtest results
        start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        end_date = datetime.datetime.now()
        
        # Mock trading results
        mock_results = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': 365
            },
            'trades_count': 150,
            'winning_trades': 85,
            'losing_trades': 65,
            'daily_returns': [0.001, -0.002, 0.003, 0.000, 0.002] * 73,  # Mock daily returns
            'cumulative_returns': [],
            'positions': [],
            'strategy_signals': {}
        }
        
        # Calculate cumulative returns
        cumulative = 1.0
        for daily_ret in mock_results['daily_returns']:
            cumulative *= (1 + daily_ret)
            mock_results['cumulative_returns'].append(cumulative - 1)
        
        mock_results['final_return'] = cumulative - 1
        
        return mock_results
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_results: Raw backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        daily_returns = backtest_results.get('daily_returns', [])
        if not daily_returns:
            return {}
        
        # Calculate basic metrics
        total_return = backtest_results.get('final_return', 0)
        
        # Sharpe ratio (simplified calculation)
        if daily_returns:
            avg_return = sum(daily_returns) / len(daily_returns)
            return_std = (sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
            sharpe_ratio = (avg_return * 252) / (return_std * (252 ** 0.5)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = backtest_results.get('cumulative_returns', [])
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0
            for ret in cumulative_returns:
                if ret > peak:
                    peak = ret
                drawdown = (ret - peak) / (1 + peak) if peak != -1 else 0
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        # Win rate
        winning_trades = backtest_results.get('winning_trades', 0)
        total_trades = backtest_results.get('trades_count', 1)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'volatility': return_std * (252 ** 0.5) if 'return_std' in locals() else 0,
            'calculated_at': datetime.datetime.now().isoformat()
        }
    
    def _summarize_factors(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize factors for AI processing."""
        summary = {
            'factor_count': len(factors),
            'factor_names': list(factors.keys()),
            'factor_types': {},
            'factor_statistics': {}
        }
        
        for factor_name, factor_data in factors.items():
            if isinstance(factor_data, dict):
                # Determine factor type
                method = factor_data.get('method', 'unknown')
                summary['factor_types'][factor_name] = method
                
                # Extract statistics
                stats = factor_data.get('statistics', {})
                if stats:
                    summary['factor_statistics'][factor_name] = stats
        
        return summary
    
    def _parse_strategy_design(self, response_content: str) -> Dict[str, Any]:
        """Parse strategy design from AI response."""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                json_content = response_content[start:end].strip()
            else:
                json_content = response_content.strip()
            
            strategy = json.loads(json_content)
            if isinstance(strategy, dict):
                return strategy
            
        except (json.JSONDecodeError, KeyError):
            self.logger.warning("Failed to parse AI response as JSON")
        
        # Fallback: create default strategy
        return {
            'name': 'simple_momentum_strategy',
            'type': 'momentum',
            'entry_rules': {
                'signal_threshold': 0.02,
                'factors_required': ['momentum', 'volume']
            },
            'exit_rules': {
                'stop_loss': -0.05,
                'take_profit': 0.10,
                'max_holding_period': 5
            },
            'position_sizing': {
                'method': 'equal_weight',
                'max_position_size': 0.1
            },
            'risk_management': {
                'max_drawdown_limit': -0.15,
                'max_leverage': 1.0
            }
        }
    
    def _validate_strategy_stability(self, backtest_results: Dict[str, Any]) -> bool:
        """Validate strategy stability from backtest results."""
        daily_returns = backtest_results.get('daily_returns', [])
        if not daily_returns or len(daily_returns) < 10:
            return False
        
        # Check for extreme outliers
        returns_sorted = sorted(daily_returns)
        q1_idx = len(returns_sorted) // 4
        q3_idx = 3 * len(returns_sorted) // 4
        
        if q1_idx < len(returns_sorted) and q3_idx < len(returns_sorted):
            q1, q3 = returns_sorted[q1_idx], returns_sorted[q3_idx]
            iqr = q3 - q1
            
            # Check for outliers beyond 3*IQR
            outlier_threshold = 3 * iqr
            outliers = [r for r in daily_returns if abs(r - (q1 + q3) / 2) > outlier_threshold]
            
            # Strategy is unstable if more than 5% outliers
            if len(outliers) / len(daily_returns) > 0.05:
                return False
        
        return True
    
    def _estimate_completion_percentage(self, state: ProcessorState) -> float:
        """Estimate backtest completion percentage."""
        status = state.get('status', 'unknown')
        
        if status == 'success':
            return 100.0
        elif status == 'error':
            return 0.0
        else:
            # Estimate based on current stage
            output_data = state.get('output_data', {})
            if 'strategy_design' in output_data:
                if 'backtest_results' in output_data:
                    if 'performance_metrics' in output_data:
                        return 90.0  # Almost complete
                    else:
                        return 70.0  # Backtest done, calculating metrics
                else:
                    return 30.0  # Strategy designed, running backtest
            else:
                return 10.0  # Just started 