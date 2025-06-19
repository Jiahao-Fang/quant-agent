"""
Factor Enhancement Library

This module provides various non-linear factor enhancement operators
supporting three data types: numpy.ndarray, pandas.Series/DataFrame, and pykx.Table.
Each function is optimized for the specific data type's performance characteristics.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Callable
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import QuantileTransformer
import warnings
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


# ================================
# 1. Z-Score Standardization
# ================================

def zscore_normalize(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'], 
                    window: Optional[int] = None) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Z-score normalization with optional rolling window
    
    Args:
        data: Input factor data
        window: Rolling window size (None for full sample)
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     if window is None:
    #         return (data - np.nanmean(data)) / np.nanstd(data)
    #     else:
    #         # Rolling z-score for numpy
    #         result = np.full_like(data, np.nan)
    #         for i in range(window-1, len(data)):
    #             subset = data[i-window+1:i+1]
    #             result[i] = (data[i] - np.nanmean(subset)) / np.nanstd(subset)
    #         return result
    
    if data_type == 'pandas':
        if window is None:
            return (data - data.mean()) / data.std()
        else:
            return (data - data.rolling(window).mean()) / data.rolling(window).std()
    
    elif data_type == 'pykx':
        
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if window is None:
            # Full sample z-score in q
            q_expr = f"update zscore: (value - avg value) % dev value from `{col_name} xcol select value:{col_name} from t"
        else:
            # Rolling z-score in q
            q_expr = f"update zscore: (value - {window} mavg value) % {window} mdev value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 2. Factor Cutting/Bucketing
# ================================

def factor_cut(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
               n_bins: int = 5,
               method: str = 'quantile') -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Factor cutting/bucketing with different methods
    
    Args:
        data: Input factor data
        n_bins: Number of bins
        method: 'quantile', 'equal_width', or 'tree'
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     if method == 'quantile':
    #         return np.digitize(data, np.nanquantile(data, np.linspace(0, 1, n_bins+1)))
    #     elif method == 'equal_width':
    #         return np.digitize(data, np.linspace(np.nanmin(data), np.nanmax(data), n_bins+1))
    #     elif method == 'tree':
    #         # Use decision tree for optimal cutting points
    #         valid_mask = ~np.isnan(data)
    #         if np.sum(valid_mask) < n_bins:
    #             return np.full_like(data, 1)
            
    #         # Create pseudo targets for tree-based cutting
    #         sorted_indices = np.argsort(data[valid_mask])
    #         pseudo_target = np.zeros(np.sum(valid_mask))
    #         bin_size = len(pseudo_target) // n_bins
    #         for i in range(n_bins):
    #             start_idx = i * bin_size
    #             end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(pseudo_target)
    #             pseudo_target[sorted_indices[start_idx:end_idx]] = i
            
    #         tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=42)
    #         tree.fit(data[valid_mask].reshape(-1, 1), pseudo_target)
            
    #         result = np.full_like(data, np.nan)
    #         result[valid_mask] = tree.predict(data[valid_mask].reshape(-1, 1))
    #         return result + 1  # 1-indexed bins
    
    if data_type == 'pandas':
        if method == 'quantile':
            return pd.qcut(data, n_bins, labels=False, duplicates='drop') + 1
        elif method == 'equal_width':
            return pd.cut(data, n_bins, labels=False) + 1
        elif method == 'tree':
            valid_data = data.dropna()
            if len(valid_data) < n_bins:
                return pd.Series(1, index=data.index)
            
            # Tree-based cutting for pandas
            sorted_data = valid_data.sort_values()
            pseudo_target = pd.Series(0, index=sorted_data.index)
            bin_size = len(sorted_data) // n_bins
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_data)
                pseudo_target.iloc[start_idx:end_idx] = i
            
            tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=42)
            tree.fit(valid_data.values.reshape(-1, 1), pseudo_target.values)
            
            result = pd.Series(np.nan, index=data.index)
            result[valid_data.index] = tree.predict(valid_data.values.reshape(-1, 1))
            return result + 1
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if method == 'quantile':
            # Quantile-based cutting in q
            q_expr = f"update bin: 1 + {n_bins} xrank value from `{col_name} xcol select value:{col_name} from t"
        elif method == 'equal_width':
            # Equal width cutting in q
            q_expr = f"update bin: 1 + `int$({n_bins}-1) * (value - min value) % max[value] - min value from `{col_name} xcol select value:{col_name} from t"
        else:
            # Fallback to quantile for tree method in q
            q_expr = f"update bin: 1 + {n_bins} xrank value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 3. Rank Transformation
# ================================

def rank_transform(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                  method: str = 'ordinal') -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Rank transformation with different tie-breaking methods
    
    Args:
        data: Input factor data
        method: 'ordinal', 'dense', 'fractional'
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     if method == 'ordinal':
    #         return stats.rankdata(data, method='ordinal', nan_policy='omit')
    #     elif method == 'dense':
    #         return stats.rankdata(data, method='dense', nan_policy='omit')
    #     elif method == 'fractional':
    #         return stats.rankdata(data, method='average', nan_policy='omit')
    
    if data_type == 'pandas':
        return data.rank(method=method, na_option='keep')
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if method in ['ordinal', 'dense']:
            q_expr = f"update rank: rank value from `{col_name} xcol select value:{col_name} from t"
        else:  # fractional
            q_expr = f"update rank: avg rank value by value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 4. Non-linear Transformations
# ================================

def nonlinear_transform(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                       transform_type: str = 'log') -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Non-linear transformations
    
    Args:
        data: Input factor data
        transform_type: 'log', 'sqrt', 'sigmoid', 'tanh', 'power'
    """
    data_type = _get_data_type(data)
    
    def _apply_transform(x, t_type):
        if t_type == 'log':
            return np.sign(x) * np.log(1 + np.abs(x))
        elif t_type == 'sqrt':
            return np.sign(x) * np.sqrt(np.abs(x))
        elif t_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif t_type == 'tanh':
            return np.tanh(x)
        elif t_type == 'power':
            return np.sign(x) * np.power(np.abs(x), 0.5)
        else:
            raise ValueError(f"Unknown transform type: {t_type}")
    
    # if data_type == 'numpy':
    #     return _apply_transform(data, transform_type)
    
    if data_type == 'pandas':
        return data.apply(lambda x: _apply_transform(x, transform_type))
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if transform_type == 'log':
            q_expr = f"update transformed: (signum value) * log 1 + abs value from `{col_name} xcol select value:{col_name} from t"
        elif transform_type == 'sqrt':
            q_expr = f"update transformed: (signum value) * sqrt abs value from `{col_name} xcol select value:{col_name} from t"
        elif transform_type == 'sigmoid':
            q_expr = f"update transformed: 1 % 1 + exp neg value from `{col_name} xcol select value:{col_name} from t"
        elif transform_type == 'tanh':
            q_expr = f"update transformed: (exp[2*value] - 1) % exp[2*value] + 1 from `{col_name} xcol select value:{col_name} from t"
        else:
            q_expr = f"update transformed: (signum value) * exp[0.5 * log abs value] from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 5. Winsorization (Outlier Treatment)
# ================================

def winsorize(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
              limits: tuple = (0.05, 0.05)) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Winsorization to handle outliers
    
    Args:
        data: Input factor data
        limits: (lower_limit, upper_limit) as percentiles
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     lower_bound = np.nanquantile(data, limits[0])
    #     upper_bound = np.nanquantile(data, 1 - limits[1])
    #     return np.clip(data, lower_bound, upper_bound)
    
    if data_type == 'pandas':
        lower_bound = data.quantile(limits[0])
        upper_bound = data.quantile(1 - limits[1])
        return data.clip(lower_bound, upper_bound)
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        lower_pct = limits[0]
        upper_pct = 1 - limits[1]
        q_expr = f"update winsorized: (value ^ ({lower_pct} wavg asc value)) & ({upper_pct} wavg asc value) from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 6. Rolling Statistical Moments
# ================================

def rolling_moments(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                   window: int = 20,
                   moment: str = 'skew') -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Rolling statistical moments (skewness, kurtosis, etc.)
    
    Args:
        data: Input factor data
        window: Rolling window size
        moment: 'skew', 'kurt', 'std_ratio'
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     result = np.full_like(data, np.nan)
    #     for i in range(window-1, len(data)):
    #         subset = data[i-window+1:i+1]
    #         if moment == 'skew':
    #             result[i] = stats.skew(subset, nan_policy='omit')
    #         elif moment == 'kurt':
    #             result[i] = stats.kurtosis(subset, nan_policy='omit')
    #         elif moment == 'std_ratio':
    #             recent_std = np.nanstd(subset[:window//2])
    #             past_std = np.nanstd(subset[window//2:])
    #             result[i] = recent_std / past_std if past_std != 0 else 1
    #     return result
    
    if data_type == 'pandas':
        if moment == 'skew':
            return data.rolling(window).skew()
        elif moment == 'kurt':
            return data.rolling(window).kurt()
        elif moment == 'std_ratio':
            recent_std = data.rolling(window//2).std()
            past_std = data.shift(window//2).rolling(window//2).std()
            return recent_std / past_std
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if moment == 'skew':
            # Approximate skewness calculation in q
            q_expr = f"update moment: {window} mavg (value - {window} mavg value) xexp 3 from `{col_name} xcol select value:{col_name} from t"
        elif moment == 'kurt':
            # Approximate kurtosis calculation in q
            q_expr = f"update moment: {window} mavg (value - {window} mavg value) xexp 4 from `{col_name} xcol select value:{col_name} from t"
        else:  # std_ratio
            half_window = window // 2
            q_expr = f"update moment: ({half_window} mdev value) % {half_window} mdev {half_window} xprev value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 7. Factor Decay (Exponential Weighting)
# ================================

def factor_decay(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                decay_factor: float = 0.94) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Exponential decay weighting of factor values
    
    Args:
        data: Input factor data
        decay_factor: Decay rate (0 < decay_factor < 1)
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     result = np.full_like(data, np.nan)
    #     result[0] = data[0]
    #     for i in range(1, len(data)):
    #         if not np.isnan(data[i]):
    #             result[i] = decay_factor * result[i-1] + (1 - decay_factor) * data[i]
    #         else:
    #             result[i] = result[i-1]
    #     return result
    
    if data_type == 'pandas':
        return data.ewm(alpha=1-decay_factor, adjust=False).mean()
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        alpha = 1 - decay_factor
        q_expr = f"update decayed: {alpha} ema value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 8. Information Coefficient Weighting
# ================================

def ic_weighting(factor_data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                returns_data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                window: int = 20) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Weight factor values by rolling Information Coefficient with returns
    
    Args:
        factor_data: Factor values
        returns_data: Forward returns
        window: Rolling window for IC calculation
    """
    data_type = _get_data_type(factor_data)
    
    # if data_type == 'numpy':
    #     result = np.full_like(factor_data, np.nan)
    #     for i in range(window, len(factor_data)):
    #         factor_subset = factor_data[i-window:i]
    #         returns_subset = returns_data[i-window:i]
            
    #         # Calculate IC (correlation)
    #         mask = ~(np.isnan(factor_subset) | np.isnan(returns_subset))
    #         if np.sum(mask) > 3:
    #             ic = np.corrcoef(factor_subset[mask], returns_subset[mask])[0, 1]
    #             result[i] = factor_data[i] * abs(ic) if not np.isnan(ic) else factor_data[i]
    #         else:
    #             result[i] = factor_data[i]
    #     return result
    
    if data_type == 'pandas':
        rolling_ic = factor_data.rolling(window).corr(returns_data).abs()
        return factor_data * rolling_ic
    
    elif data_type == 'pykx':
        # Simplified IC weighting for pykx
        factor_col = factor_data.columns[0] if len(factor_data.columns) > 0 else 'factor'
        returns_col = returns_data.columns[0] if len(returns_data.columns) > 0 else 'returns'
        
        q_expr = f"update weighted: factor * abs {window} mavg (factor - {window} mavg factor) * (returns - {window} mavg returns) from t"
        
        # Combine data for correlation calculation
        combined_data = kx.q("t1 lj `time xkey t2", t1=factor_data, t2=returns_data)
        return kx.q(q_expr, t=combined_data)


# ================================
# 9. Quantile-based Normalization
# ================================

def quantile_normalize(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                      reference_dist: str = 'normal') -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Transform data to follow a reference distribution
    
    Args:
        data: Input factor data
        reference_dist: 'normal', 'uniform'
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     transformer = QuantileTransformer(
    #         output_distribution=reference_dist,
    #         random_state=42
    #     )
    #     valid_mask = ~np.isnan(data)
    #     result = np.full_like(data, np.nan)
    #     if np.sum(valid_mask) > 1:
    #         result[valid_mask] = transformer.fit_transform(
    #             data[valid_mask].reshape(-1, 1)
    #         ).flatten()
    #     return result
    
    if data_type == 'pandas':
        transformer = QuantileTransformer(
            output_distribution=reference_dist,
            random_state=42
        )
        valid_data = data.dropna()
        if len(valid_data) > 1:
            transformed = transformer.fit_transform(valid_data.values.reshape(-1, 1))
            result = pd.Series(np.nan, index=data.index)
            result[valid_data.index] = transformed.flatten()
            return result
        else:
            return data
    
    elif data_type == 'pykx':
        col_name = data.columns[0] if len(data.columns) > 0 else 'value'
        if reference_dist == 'uniform':
            q_expr = f"update normalized: (rank value) % count value from `{col_name} xcol select value:{col_name} from t"
        else:  # normal approximation
            q_expr = f"update normalized: (value - avg value) % dev value from `{col_name} xcol select value:{col_name} from t"
        
        return kx.q(q_expr, t=data)


# ================================
# 10. Regime-based Transformation
# ================================

def regime_transform(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                    regime_indicator: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                    n_regimes: int = 3) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Apply different transformations based on market regime
    
    Args:
        data: Input factor data
        regime_indicator: Market regime indicator (e.g., volatility, trend)
        n_regimes: Number of regimes
    """
    data_type = _get_data_type(data)
    
    # if data_type == 'numpy':
    #     # Determine regimes based on quantiles of regime indicator
    #     regime_cuts = np.nanquantile(regime_indicator, np.linspace(0, 1, n_regimes+1))
    #     regimes = np.digitize(regime_indicator, regime_cuts) - 1
        
    #     result = np.full_like(data, np.nan)
    #     for regime in range(n_regimes):
    #         mask = regimes == regime
    #         if np.sum(mask) > 0:
    #             regime_data = data[mask]
    #             # Apply different transformation per regime
    #             if regime == 0:  # Low regime - use raw values
    #                 result[mask] = regime_data
    #             elif regime == 1:  # Medium regime - use rank
    #                 result[mask] = stats.rankdata(regime_data, nan_policy='omit')
    #             else:  # High regime - use z-score
    #                 result[mask] = (regime_data - np.nanmean(regime_data)) / np.nanstd(regime_data)
        
    #     return result
    
    if data_type == 'pandas':
        regime_cuts = regime_indicator.quantile(np.linspace(0, 1, n_regimes+1))
        regimes = pd.cut(regime_indicator, regime_cuts, labels=False, include_lowest=True)
        
        result = pd.Series(np.nan, index=data.index)
        for regime in range(n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                regime_data = data[mask]
                if regime == 0:
                    result[mask] = regime_data
                elif regime == 1:
                    result[mask] = regime_data.rank()
                else:
                    result[mask] = (regime_data - regime_data.mean()) / regime_data.std()
        
        return result
    
    elif data_type == 'pykx':
        # Simplified regime-based transformation for pykx
        data_col = data.columns[0] if len(data.columns) > 0 else 'value'
        regime_col = regime_indicator.columns[0] if len(regime_indicator.columns) > 0 else 'regime'
        
        q_expr = f"""
        update transformed: ?[
            regime_bin < 1; value;
            regime_bin < 2; rank value;
            (value - avg value) % dev value
        ] by regime_bin: {n_regimes} xrank {regime_col} from t
        """
        
        # Combine data
        combined_data = kx.q("t1 lj `time xkey t2", t1=data, t2=regime_indicator)
        return kx.q(q_expr, t=combined_data)


# ================================
# Main Enhancement Function
# ================================

def enhance_factor(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                  enhancement_type: str,
                  **kwargs) -> Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table']:
    """
    Main factor enhancement function that routes to specific enhancement methods
    
    Args:
        data: Input factor data
        enhancement_type: Type of enhancement to apply
        **kwargs: Additional parameters for specific enhancement methods
    
    Available enhancement types:
        - 'zscore': Z-score normalization
        - 'cut': Factor cutting/bucketing
        - 'rank': Rank transformation
        - 'nonlinear': Non-linear transformations
        - 'winsorize': Outlier treatment
        - 'moments': Rolling statistical moments
        - 'decay': Exponential decay weighting
        - 'ic_weight': Information coefficient weighting
        - 'quantile_norm': Quantile-based normalization
        - 'regime': Regime-based transformation
    """
    enhancement_map = {
        'zscore': zscore_normalize,
        'cut': factor_cut,
        'rank': rank_transform,
        'nonlinear': nonlinear_transform,
        'winsorize': winsorize,
        'moments': rolling_moments,
        'decay': factor_decay,
        'ic_weight': ic_weighting,
        'quantile_norm': quantile_normalize,
        'regime': regime_transform
    }
    
    if enhancement_type not in enhancement_map:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")
    
    return enhancement_map[enhancement_type](data, **kwargs)


# ================================
# Batch Enhancement
# ================================

def batch_enhance(data: Union[np.ndarray, pd.Series, pd.DataFrame, 'kx.Table'],
                 enhancement_pipeline: list) -> dict:
    """
    Apply multiple enhancements in sequence
    
    Args:
        data: Input factor data
        enhancement_pipeline: List of (enhancement_type, kwargs) tuples
    
    Returns:
        Dictionary with enhancement results
    """
    results = {'original': data}
    
    current_data = data
    for enhancement_type, kwargs in enhancement_pipeline:
        current_data = enhance_factor(current_data, enhancement_type, **kwargs)
        results[enhancement_type] = current_data
    
    return results
