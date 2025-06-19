"""
Quantitative Factor Backtesting Module

This module provides comprehensive backtesting functionality for quantitative factors,
supporting multiple data types (numpy, pandas, pykx) and extensive evaluation metrics
that quantitative analysts care about.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import warnings
from datetime import datetime, timedelta
import re
import pykx as kx


# ================================
# Utility Functions
# ================================

def _get_data_type(data):
    """Detect data type and return appropriate handler"""
    # if isinstance(data, np.ndarray):
    #     return 'numpy'
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return 'pandas'
    elif isinstance(data, kx.Table):
        return 'pykx'
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _parse_walkahead(walkahead: str) -> int:
    """
    Parse walkahead string to nanoseconds
    
    Args:
        walkahead: String like '5s', '30m', '1h', '1d'
    
    Returns:
        Number of nanoseconds to shift forward
    """
    pattern = r'^(\d+)([smhd])$'
    match = re.match(pattern, walkahead.lower())
    
    if not match:
        raise ValueError(f"Invalid walkahead format: {walkahead}. Use format like '5s', '30m', '1h', '1d'")
    
    value, unit = int(match.group(1)), match.group(2)
    
    # Convert to nanoseconds
    multipliers = {
        's': 1_000_000_000,     # 1 second = 1e9 nanoseconds
        'm': 60_000_000_000,    # 1 minute = 60 * 1e9 nanoseconds
        'h': 3_600_000_000_000, # 1 hour = 3600 * 1e9 nanoseconds
        'd': 86_400_000_000_000 # 1 day = 86400 * 1e9 nanoseconds
    }
    
    return value * multipliers[unit]


def _create_shifted_target(data, target_col: str, timestamp_col: str, walkahead: str, data_type: str):
    """Create forward-shifted target using asof merge with walkahead offset"""
    shift_nanoseconds = _parse_walkahead(walkahead)
    
    if data_type == 'pandas':
        # Create future timestamps for asof merge
        data_sorted = data.sort_values(timestamp_col).copy()
        
        # Create target data with future timestamps
        target_data = data_sorted[[timestamp_col, target_col]].copy()
        target_data[timestamp_col] = target_data[timestamp_col] + shift_nanoseconds
        target_data = target_data.rename(columns={target_col: 'shifted_target'})
        
        # Use asof merge to align future targets with current timestamps
        result = pd.merge_asof(
            data_sorted.sort_values(timestamp_col),
            target_data.sort_values(timestamp_col),
            on=timestamp_col,
            direction='forward'
        )
        
        return result
    # TO_DO: this is low efficiency, need to optimize
    # elif data_type == 'numpy':
    #     # For numpy, convert to pandas temporarily for asof merge
    #     # Assume first column is timestamp, last column is target
    #     df_temp = pd.DataFrame(data)
    #     df_temp.columns = [timestamp_col if i == 0 else f'feature_{i}' if i < data.shape[1]-1 else target_col 
    #                       for i in range(data.shape[1])]
        
    #     # Apply pandas logic
    #     result_df = _create_shifted_target(df_temp, target_col, timestamp_col, walkahead, 'pandas')
    #     return result_df.values
        
    elif data_type == 'pykx':
        # Create target data with future timestamps
        target_data = data.copy()
        target_data[timestamp_col] = target_data[timestamp_col] + shift_nanoseconds
        target_data = target_data[[timestamp_col, target_col]]
        target_data = target_data.rename(columns={target_col: 'shifted_target'})
        
        # Use asof merge to align future targets with current timestamps
        # TO DO: check direction of merge
        result = data.merge_asof(
            target_data,
            on=timestamp_col
        )
        
        return result

# ================================
# Model Training and Prediction
# ================================

def _fit_model(X: np.ndarray, y: np.ndarray, model_type: str, **model_kwargs):
    """Fit regression model based on type"""
    models = {
        'linear': LinearRegression(),
        'lasso': Lasso(alpha=model_kwargs.get('alpha', 1.0)),
        'ridge': Ridge(alpha=model_kwargs.get('alpha', 1.0)),
        'logistic': LogisticRegression(max_iter=1000)
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = models[model_type]
    
    # Handle NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    if np.sum(valid_mask) < 10:  # Need minimum samples
        return None, None
    
    X_clean, y_clean = X[valid_mask], y[valid_mask]
    
    try:
        model.fit(X_clean, y_clean)
        predictions = np.full(len(X), np.nan)
        predictions[valid_mask] = model.predict(X_clean)
        return model, predictions
    except Exception as e:
        warnings.warn(f"Model fitting failed: {str(e)}")
        return None, None


# ================================
# Comprehensive Evaluation Metrics
# ================================

class QuantEvaluator:
    """Comprehensive quantitative evaluation metrics"""
    
    @staticmethod
    def information_coefficient(factor: np.ndarray, returns: np.ndarray) -> float:
        """Calculate Information Coefficient (IC)"""
        valid_mask = ~(np.isnan(factor) | np.isnan(returns))
        if np.sum(valid_mask) < 3:
            return np.nan
        return np.corrcoef(factor[valid_mask], returns[valid_mask])[0, 1]
    
    @staticmethod
    def rank_ic(factor: np.ndarray, returns: np.ndarray) -> float:
        """Calculate Rank Information Coefficient"""
        valid_mask = ~(np.isnan(factor) | np.isnan(returns))
        if np.sum(valid_mask) < 3:
            return np.nan
        
        factor_ranks = stats.rankdata(factor[valid_mask])
        returns_ranks = stats.rankdata(returns[valid_mask])
        return np.corrcoef(factor_ranks, returns_ranks)[0, 1]
    
    @staticmethod
    def information_ratio(ic_series: np.ndarray) -> float:
        """Calculate Information Ratio (IR)"""
        if len(ic_series) < 2:
            return np.nan
        return np.nanmean(ic_series) / np.nanstd(ic_series)
    
    @staticmethod
    def hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate hit rate (directional accuracy)"""
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if np.sum(valid_mask) < 3:
            return np.nan
        
        pred_direction = np.sign(predictions[valid_mask])
        actual_direction = np.sign(actuals[valid_mask])
        return np.mean(pred_direction == actual_direction)
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return np.nan
        excess_returns = returns - risk_free_rate
        return np.nanmean(excess_returns) / np.nanstd(excess_returns)
    
    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) < 2:
            return np.nan
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.nanmin(drawdown)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) < 2:
            return np.nan
        
        cumulative_returns = np.cumprod(1 + returns)
        annual_return = np.nanmean(returns) * 252  # Assume daily data
        max_dd = QuantEvaluator.max_drawdown(cumulative_returns)
        
        return annual_return / abs(max_dd) if max_dd != 0 else np.nan
    
    @staticmethod
    def ic_decay(factor: np.ndarray, returns: np.ndarray, max_lag: int = 10) -> np.ndarray:
        """Calculate IC decay over different forward periods"""
        ic_decay_values = []
        
        for lag in range(1, max_lag + 1):
            if lag < len(returns):
                shifted_returns = np.roll(returns, -lag)
                shifted_returns[-lag:] = np.nan
                ic = QuantEvaluator.information_coefficient(factor, shifted_returns)
                ic_decay_values.append(ic)
            else:
                ic_decay_values.append(np.nan)
        
        return np.array(ic_decay_values)
    
    @staticmethod
    def factor_turnover(factor: np.ndarray, window: int = 20) -> float:
        """Calculate factor turnover (stability)"""
        if len(factor) < window * 2:
            return np.nan
        
        turnover_values = []
        for i in range(window, len(factor) - window):
            prev_ranks = stats.rankdata(factor[i-window:i])
            curr_ranks = stats.rankdata(factor[i:i+window])
            correlation = np.corrcoef(prev_ranks, curr_ranks)[0, 1]
            turnover_values.append(1 - correlation)
        
        return np.nanmean(turnover_values)
    
    @staticmethod
    def quantile_analysis(factor: np.ndarray, returns: np.ndarray, n_quantiles: int = 5) -> Dict[str, float]:
        """Analyze returns by factor quantiles"""
        # 数据清洗
        valid_mask = ~(np.isnan(factor) | np.isnan(returns))
        if np.sum(valid_mask) < n_quantiles * 2:
            return {}
        
        factor_clean = factor[valid_mask]
        returns_clean = returns[valid_mask]
        
        # 分位数划分
        try:
            quantile_labels = pd.qcut(factor_clean, n_quantiles, labels=False, duplicates='drop')
        except:
            return {}
        
        # 计算各分位数表现
        results = {}
        for q in range(n_quantiles):
            mask = quantile_labels == q
            if np.sum(mask) > 0:
                q_returns = returns_clean[mask]
                results[f'Q{q+1}_mean_return'] = np.nanmean(q_returns)
                results[f'Q{q+1}_sharpe'] = QuantEvaluator.sharpe_ratio(q_returns)
        
        # 多空策略
        if f'Q{n_quantiles}_mean_return' in results and 'Q1_mean_return' in results:
            results['long_short_return'] = (results[f'Q{n_quantiles}_mean_return'] - 
                                        results['Q1_mean_return'])
        
        return results


# ================================
# Main Backtesting Class
# ================================

class QuantBacktester:
    """
    Comprehensive quantitative factor backtesting framework
    """
    
    def __init__(self,
                 feature_cols: List[str],
                 target_col: str,
                 timestamp_col: str = 'recvTimestamp',
                 walkahead: str = '1s',
                 model_type: str = 'linear',
                 test_size: float = 0.3,
                 cv_folds: int = 5,
                 metrics: Optional[List[str]] = None):
        """
        Initialize backtester
        
        Args:
            feature_cols: List of feature column names
            target_col: Target column name
            timestamp_col: Timestamp column name
            walkahead: Forward shift for target (e.g., '5s', '30m', '1h')
            model_type: 'linear', 'lasso', 'ridge', 'logistic'
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            metrics: List of metrics to calculate (None for all)
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.walkahead = walkahead
        self.model_type = model_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        
        # Store feature names for later use
        self.feature_names = feature_cols.copy()
        
        # Available metrics
        self.all_metrics = [
            'r2', 'mse', 'mae', 'ic', 'rank_ic', 'ir', 'hit_rate',
            'sharpe', 'max_drawdown', 'calmar', 'ic_decay', 'turnover',
            'quantile_analysis', 'rolling_ic', 'factor_correlation'
        ]
        
        self.metrics = metrics if metrics is not None else self.all_metrics
        self.evaluator = QuantEvaluator()
        
    def _prepare_data(self, data, data_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for backtesting"""
        
        # Create shifted target
        data_with_target = _create_shifted_target(
            data, self.target_col, self.timestamp_col, self.walkahead, data_type
        )
        
        if data_type == 'pandas':
            # Extract features and target, ensure target is last column
            X = data_with_target[self.feature_cols].values
            y = data_with_target['shifted_target'].values
            timestamps = data_with_target[self.timestamp_col].values
            
            # Create combined array with target as last column
            combined = np.column_stack([X, y])
            feature_names = self.feature_names.copy()
            
        # elif data_type == 'numpy':
        #     # Assume data_with_target already has target as last column
        #     combined = data_with_target
        #     X = combined[:, 1:-1]  # Skip timestamp (first col) and target (last col)
        #     y = combined[:, -1]    # Last column is target
        #     timestamps = combined[:, 0]  # First column is timestamp
            
        #     # Generate feature names if not provided
        #     feature_names = self.feature_names if len(self.feature_names) == X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
        # TO_DO: this is low efficiency, need to optimize
        elif data_type == 'pykx':
            # Convert to pandas for easier processing
            df = data_with_target.pd()
            X = df[self.feature_cols].values
            y = df['shifted_target'].values
            timestamps = df[self.timestamp_col].values
            
            # Create combined array with target as last column
            combined = np.column_stack([X, y])
            feature_names = self.feature_names.copy()
        
        return combined, y, timestamps, feature_names
    
    def _time_series_split(self, combined_data: np.ndarray, timestamps: np.ndarray):
        """Create time-aware train/test split"""
        n_samples = len(combined_data)
        split_idx = int(n_samples * (1 - self.test_size))
        
        train_data = combined_data[:split_idx]
        test_data = combined_data[split_idx:]
        timestamps_train = timestamps[:split_idx]
        timestamps_test = timestamps[split_idx:]
        
        # Extract X and y from combined data
        X_train = train_data[:, :-1]  # All columns except last (target)
        y_train = train_data[:, -1]   # Last column (target)
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test
    
    def _calculate_rolling_metrics(self, factor: np.ndarray, returns: np.ndarray, 
                                 window: int = 252) -> Dict[str, np.ndarray]:
        """Calculate rolling metrics over time"""
        rolling_ic = []
        rolling_rank_ic = []
        
        for i in range(window, len(factor)):
            f_window = factor[i-window:i]
            r_window = returns[i-window:i]
            
            ic = self.evaluator.information_coefficient(f_window, r_window)
            rank_ic = self.evaluator.rank_ic(f_window, r_window)
            
            rolling_ic.append(ic)
            rolling_rank_ic.append(rank_ic)
        
        return {
            'rolling_ic': np.array(rolling_ic),
            'rolling_rank_ic': np.array(rolling_rank_ic)
        }
    
    def backtest(self, data: Union[np.ndarray, pd.DataFrame, kx.Table],
                benchmark_factors: Optional[Dict[str, np.ndarray]] = None,
                **model_kwargs) -> Dict[str, Any]:
        """
        Run comprehensive backtesting
        
        Args:
            data: Input data (numpy array, pandas DataFrame, or pykx Table)
            benchmark_factors: Dictionary of benchmark factors for correlation analysis
            **model_kwargs: Additional model parameters
        
        Returns:
            Dictionary containing all evaluation results
        """
        data_type = _get_data_type(data)
        
        # Prepare data
        combined_data, y_full, timestamps, feature_names = self._prepare_data(data, data_type)
        
        # Time series split
        X_train, X_test, y_train, y_test, ts_train, ts_test = self._time_series_split(combined_data, timestamps)
        
        # Fit model
        model, predictions = _fit_model(X_test, y_test, self.model_type, **model_kwargs)
        
        if model is None:
            return {'error': 'Model fitting failed'}
        
        # Initialize results
        results = {
            'model_type': self.model_type,
            'walkahead': self.walkahead,
            'n_features': X_test.shape[1],
            'feature_names': feature_names,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_params': model.get_params() if hasattr(model, 'get_params') else {}
        }
        
        # Basic regression metrics
        valid_predictions = predictions[~np.isnan(predictions)]
        valid_targets = y_test[~np.isnan(predictions)]
        
        if len(valid_predictions) > 0:
            if 'r2' in self.metrics:
                results['r2'] = r2_score(valid_targets, valid_predictions)
            
            if 'mse' in self.metrics:
                results['mse'] = mean_squared_error(valid_targets, valid_predictions)
            
            if 'mae' in self.metrics:
                results['mae'] = mean_absolute_error(valid_targets, valid_predictions)
        
        # Factor-specific metrics
        if 'ic' in self.metrics:
            results['ic'] = self.evaluator.information_coefficient(predictions, y_test)
        
        if 'rank_ic' in self.metrics:
            results['rank_ic'] = self.evaluator.rank_ic(predictions, y_test)
        
        if 'hit_rate' in self.metrics:
            results['hit_rate'] = self.evaluator.hit_rate(predictions, y_test)
        
        # Risk-adjusted metrics
        if 'sharpe' in self.metrics:
            results['sharpe'] = self.evaluator.sharpe_ratio(y_test)
        
        if 'max_drawdown' in self.metrics:
            cumulative_returns = np.cumprod(1 + y_test[~np.isnan(y_test)])
            results['max_drawdown'] = self.evaluator.max_drawdown(cumulative_returns)
        
        if 'calmar' in self.metrics:
            results['calmar'] = self.evaluator.calmar_ratio(y_test)
        
        # Advanced metrics
        if 'ic_decay' in self.metrics:
            results['ic_decay'] = self.evaluator.ic_decay(predictions, y_test)
        
        if 'turnover' in self.metrics:
            results['turnover'] = self.evaluator.factor_turnover(predictions)
        
        if 'quantile_analysis' in self.metrics:
            results['quantile_analysis'] = self.evaluator.quantile_analysis(predictions, y_test)
        
        # Rolling metrics
        if 'rolling_ic' in self.metrics:
            rolling_metrics = self._calculate_rolling_metrics(predictions, y_test)
            results.update(rolling_metrics)
            
            # Information Ratio from rolling IC
            if 'ir' in self.metrics:
                results['ir'] = self.evaluator.information_ratio(rolling_metrics['rolling_ic'])
        
        # Benchmark correlation analysis
        if 'factor_correlation' in self.metrics and benchmark_factors:
            correlations = {}
            for name, benchmark in benchmark_factors.items():
                if len(benchmark) == len(predictions):
                    corr = self.evaluator.information_coefficient(predictions, benchmark)
                    correlations[f'corr_with_{name}'] = corr
            results['factor_correlations'] = correlations
        
        # Cross-validation results
        if self.cv_folds > 1:
            cv_results = self._cross_validate(X_train, y_train, **model_kwargs)
            results['cv_results'] = cv_results
        
        # Model coefficients with feature names
        if hasattr(model, 'coef_'):
            results['feature_importance'] = {
                feature_names[i]: coef for i, coef in enumerate(model.coef_)
            }
        
        return results
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, **model_kwargs) -> Dict[str, float]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = []
        cv_ics = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model, predictions = _fit_model(X_val_cv, y_val_cv, self.model_type, **model_kwargs)
            
            if model is not None:
                score = r2_score(y_val_cv[~np.isnan(predictions)], 
                               predictions[~np.isnan(predictions)])
                ic = self.evaluator.information_coefficient(predictions, y_val_cv)
                
                cv_scores.append(score)
                cv_ics.append(ic)
        
        return {
            'cv_r2_mean': np.nanmean(cv_scores),
            'cv_r2_std': np.nanstd(cv_scores),
            'cv_ic_mean': np.nanmean(cv_ics),
            'cv_ic_std': np.nanstd(cv_ics)
        }


# ================================
# Convenience Functions
# ================================

def quick_backtest(data: Union[np.ndarray, pd.DataFrame, kx.Table],
                  feature_cols: List[str],
                  target_col: str,
                  **kwargs) -> Dict[str, Any]:
    """
    Quick backtesting with default parameters
    
    Args:
        data: Input data
        feature_cols: Feature column names
        target_col: Target column name
        **kwargs: Additional parameters for QuantBacktester
    
    Returns:
        Backtesting results
    """
    backtester = QuantBacktester(feature_cols, target_col, **kwargs)
    return backtester.backtest(data)


def batch_backtest(datasets: Dict[str, Union[np.ndarray, pd.DataFrame, kx.Table]],
                  feature_cols: List[str],
                  target_col: str,
                  **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run backtesting on multiple datasets
    
    Args:
        datasets: Dictionary of dataset_name -> data
        feature_cols: Feature column names
        target_col: Target column name
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with results for each dataset
    """
    results = {}
    
    for dataset_name, data in datasets.items():
        try:
            backtester = QuantBacktester(feature_cols, target_col, **kwargs)
            results[dataset_name] = backtester.backtest(data)
        except Exception as e:
            results[dataset_name] = {'error': str(e)}
    
    return results
