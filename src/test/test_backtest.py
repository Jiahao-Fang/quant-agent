"""
Unit tests for backtest.py module

This test suite creates synthetic KDB-like data using pykx to test
all functions in the backtest module.
"""

import unittest
import numpy as np
import pandas as pd
import pykx as kx
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import (
    _get_data_type, _parse_walkahead, _create_shifted_target, _fit_model,
    QuantEvaluator, QuantBacktester, quick_backtest, batch_backtest
)


class TestDataGenerator:
    """Generate synthetic KDB-like data for testing"""
    
    @staticmethod
    def create_test_data_pykx(n_samples: int = 1000, 
                             start_time: datetime = datetime(2025, 2, 1, 9, 0)) -> kx.Table:
        """
        Create synthetic tick data similar to KDB structure
        
        Args:
            n_samples: Number of data points
            start_time: Start timestamp
            
        Returns:
            pykx Table with tick data
        """
        # Generate data directly using pykx
        np.random.seed(42)
        
        # Generate timestamps using q's random function
        timestamps = kx.q.asc(kx.random.random(n_samples, kx.q('1D')))
        timestamps = timestamps + kx.DateAtom(start_time)

        # Generate synthetic features using pykx random functions
        bid_qty = kx.random.random(n_samples, kx.FloatAtom(100.0))  # exponential distribution
        ask_qty = kx.random.random(n_samples, kx.FloatAtom(100.0))
        book_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        
        price_changes = kx.random.random(n_samples, kx.FloatAtom(0.001))  # normal distribution
        price_momentum = kx.q.sums(price_changes)
        
        volumes = kx.random.random(n_samples, kx.FloatAtom(1000.0))
        vwap_dev = kx.random.random(n_samples, kx.FloatAtom(0.01))
        
        spreads = kx.random.random(n_samples, kx.FloatAtom(0.001))
        spread_feature = (spreads - kx.q.avg(spreads)) / kx.q.dev(spreads)
        
        # Create synthetic target with noise
        noise = kx.random.random(n_samples, kx.FloatAtom(0.002))
        future_returns = (
            0.3 * book_imbalance + 
            0.2 * price_momentum +
            -0.1 * vwap_dev +
            -0.15 * spread_feature +
            noise
        )
        
        # Create kdb table directly
        return kx.Table(data={
            'recvTimestamp': timestamps,
            'symbol': kx.random.random(n_samples, ['BTCUSDT']),
            'exchange': kx.random.random(n_samples, ['ST_BNS']),
            'book_imbalance': book_imbalance,
            'price_momentum': price_momentum, 
            'vwap_deviation': vwap_dev,
            'spread_feature': spread_feature,
            'bid_qty_0': bid_qty,
            'ask_qty_0': ask_qty,
            'volume': volumes,
            'future_returns': future_returns
        })
    @staticmethod
    def create_test_data_pandas(n_samples: int = 1000) -> pd.DataFrame:
        """Create pandas DataFrame with similar structure"""
        # Generate timestamps in nanoseconds
        start_time = pd.Timestamp('2025-02-01 09:00:00').value  # nanoseconds
        timestamps = [start_time + i * 1_000_000_000 for i in range(n_samples)]
        
        # Generate same features as pykx version
        np.random.seed(42)
        bid_qty = np.random.exponential(100, n_samples)
        ask_qty = np.random.exponential(100, n_samples)
        book_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        
        price_changes = np.random.normal(0, 0.001, n_samples)
        price_momentum = np.cumsum(price_changes)
        
        volumes = np.random.exponential(1000, n_samples)
        vwap_dev = np.random.normal(0, 0.01, n_samples)
        
        spreads = np.random.exponential(0.001, n_samples)
        spread_feature = (spreads - np.mean(spreads)) / np.std(spreads)
        
        noise = np.random.normal(0, 0.002, n_samples)
        future_returns = (
            0.3 * book_imbalance + 
            0.2 * price_momentum + 
            -0.1 * vwap_dev + 
            -0.15 * spread_feature + 
            noise
        )
        
        df = pd.DataFrame({
            'recvTimestamp': timestamps,
            'symbol': ['BTCUSDT'] * n_samples,
            'exchange': ['ST_BNS'] * n_samples,
            'book_imbalance': book_imbalance,
            'price_momentum': price_momentum,
            'vwap_deviation': vwap_dev,
            'spread_feature': spread_feature,
            'bid_qty_0': bid_qty,
            'ask_qty_0': ask_qty,
            'volume': volumes,
            'future_returns': future_returns
        })
        
        return df.sort_values('recvTimestamp')
    
    # @staticmethod
    # def create_test_data_numpy(n_samples: int = 1000) -> np.ndarray:
    #     """Create numpy array with similar structure"""
    #     df = TestDataGenerator.create_test_data_pandas(n_samples)
    #     return df.values


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_get_data_type(self):
        """Test data type detection"""
        # # Test numpy
        # np_data = np.random.rand(100, 5)
        # self.assertEqual(_get_data_type(np_data), 'numpy')
        
        # Test pandas
        pd_data = pd.DataFrame(np.random.rand(100, 5))
        self.assertEqual(_get_data_type(pd_data), 'pandas')
        
        # Test pykx
        kx_data = TestDataGenerator.create_test_data_pykx(100)
        self.assertEqual(_get_data_type(kx_data), 'pykx')
    
    def test_parse_walkahead(self):
        """Test walkahead string parsing"""
        # Test seconds
        self.assertEqual(_parse_walkahead('5s'), 5_000_000_000)
        self.assertEqual(_parse_walkahead('30s'), 30_000_000_000)
        
        # Test minutes
        self.assertEqual(_parse_walkahead('1m'), 60_000_000_000)
        self.assertEqual(_parse_walkahead('15m'), 15 * 60_000_000_000)
        
        # Test hours
        self.assertEqual(_parse_walkahead('1h'), 3_600_000_000_000)
        
        # Test days
        self.assertEqual(_parse_walkahead('1d'), 86_400_000_000_000)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            _parse_walkahead('invalid')
        
        with self.assertRaises(ValueError):
            _parse_walkahead('5x')


class TestShiftedTarget(unittest.TestCase):
    """Test target shifting functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.pykx_data = TestDataGenerator.create_test_data_pykx(100)
        self.pandas_data = TestDataGenerator.create_test_data_pandas(100)
        # self.numpy_data = TestDataGenerator.create_test_data_numpy(100)
        
    def test_create_shifted_target_pandas(self):
        """Test target shifting for pandas data"""
        result = _create_shifted_target(
            self.pandas_data,
            'future_returns',
            'recvTimestamp',
            '5s',
            'pandas'
        )
        
        # Check that shifted_target column exists
        self.assertIn('shifted_target', result.columns)
        
        # Check that most values are not NaN (except possibly at the end)
        non_nan_count = result['shifted_target'].notna().sum()
        self.assertGreater(non_nan_count, 50)  # Should have plenty of valid values
        
        # Check that result is sorted by timestamp
        self.assertTrue(result['recvTimestamp'].is_monotonic_increasing)
    
    # def test_create_shifted_target_numpy(self):
    #     """Test target shifting for numpy data"""
    #     result = _create_shifted_target(
    #         self.numpy_data,
    #         'future_returns',
    #         'recvTimestamp',
    #         '5s',
    #         'numpy'
    #     )
        
    #     # Check that result has one more column (shifted_target)
    #     self.assertEqual(result.shape[1], self.numpy_data.shape[1])
        
    #     # Check that we have a 2D array
    #     self.assertEqual(len(result.shape), 2)
    
    def test_create_shifted_target_pykx(self):
        """Test target shifting for pykx data"""
        result = _create_shifted_target(
            self.pykx_data,
            'future_returns',
            'recvTimestamp',
            '5s',
            'pykx'
        )
        
        # Check that result is a pykx table
        self.assertIsInstance(result, kx.Table)
        
        # Check that shifted_target column exists
        columns = result.columns
        self.assertIn('shifted_target', columns)


class TestModelFitting(unittest.TestCase):
    """Test model fitting functionality"""
    
    def setUp(self):
        """Set up test data for model fitting"""
        np.random.seed(42)
        self.X = np.random.randn(100, 4)
        self.y = (
            0.5 * self.X[:, 0] + 
            0.3 * self.X[:, 1] - 
            0.2 * self.X[:, 2] + 
            0.1 * self.X[:, 3] + 
            np.random.normal(0, 0.1, 100)
        )
    
    def test_fit_linear_model(self):
        """Test linear regression fitting"""
        model, predictions = _fit_model(self.X, self.y, 'linear')
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.y))
        
        # Check that most predictions are not NaN
        non_nan_count = np.sum(~np.isnan(predictions))
        self.assertGreater(non_nan_count, 80)
    
    def test_fit_lasso_model(self):
        """Test Lasso regression fitting"""
        model, predictions = _fit_model(self.X, self.y, 'lasso', alpha=0.1)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_fit_ridge_model(self):
        """Test Ridge regression fitting"""
        model, predictions = _fit_model(self.X, self.y, 'ridge', alpha=0.1)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_fit_logistic_model(self):
        """Test Logistic regression fitting"""
        # Create binary target
        y_binary = (self.y > np.median(self.y)).astype(int)
        model, predictions = _fit_model(self.X, y_binary, 'logistic')
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(y_binary))
    
    def test_fit_invalid_model(self):
        """Test invalid model type"""
        with self.assertRaises(ValueError):
            _fit_model(self.X, self.y, 'invalid_model')
    
    def test_fit_insufficient_data(self):
        """Test with insufficient data"""
        X_small = self.X[:5]
        y_small = self.y[:5]
        
        model, predictions = _fit_model(X_small, y_small, 'linear')
        self.assertIsNone(model)
        self.assertIsNone(predictions)


class TestQuantEvaluator(unittest.TestCase):
    """Test QuantEvaluator metrics"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.factor = np.random.randn(100)
        self.returns = 0.5 * self.factor + np.random.normal(0, 0.5, 100)
    
    def test_information_coefficient(self):
        """Test IC calculation"""
        ic = QuantEvaluator.information_coefficient(self.factor, self.returns)
        
        self.assertIsInstance(ic, float)
        self.assertFalse(np.isnan(ic))
        self.assertGreater(ic, 0)  # Should be positive due to correlation
        self.assertLessEqual(abs(ic), 1)  # Correlation should be between -1 and 1
    
    def test_rank_ic(self):
        """Test Rank IC calculation"""
        rank_ic = QuantEvaluator.rank_ic(self.factor, self.returns)
        
        self.assertIsInstance(rank_ic, float)
        self.assertFalse(np.isnan(rank_ic))
        self.assertLessEqual(abs(rank_ic), 1)
    
    def test_information_ratio(self):
        """Test Information Ratio calculation"""
        ic_series = np.random.normal(0.1, 0.05, 50)
        ir = QuantEvaluator.information_ratio(ic_series)
        
        self.assertIsInstance(ir, float)
        self.assertFalse(np.isnan(ir))
    
    def test_hit_rate(self):
        """Test hit rate calculation"""
        predictions = np.random.randn(100)
        actuals = np.sign(predictions) + np.random.normal(0, 0.1, 100)
        
        hit_rate = QuantEvaluator.hit_rate(predictions, actuals)
        
        self.assertIsInstance(hit_rate, float)
        self.assertFalse(np.isnan(hit_rate))
        self.assertGreaterEqual(hit_rate, 0)
        self.assertLessEqual(hit_rate, 1)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = np.random.normal(0.01, 0.1, 100)
        sharpe = QuantEvaluator.sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        cumulative_returns = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        max_dd = QuantEvaluator.max_drawdown(cumulative_returns)
        
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative
    
    def test_ic_decay(self):
        """Test IC decay calculation"""
        ic_decay = QuantEvaluator.ic_decay(self.factor, self.returns, max_lag=5)
        
        self.assertIsInstance(ic_decay, np.ndarray)
        self.assertEqual(len(ic_decay), 5)
    
    def test_factor_turnover(self):
        """Test factor turnover calculation"""
        factor = np.random.randn(100)
        turnover = QuantEvaluator.factor_turnover(factor, window=20)
        
        self.assertIsInstance(turnover, float)
        self.assertGreaterEqual(turnover, 0)
        self.assertLessEqual(turnover, 2)  # Turnover is typically 0-2
    
    def test_quantile_analysis(self):
        """Test quantile analysis"""
        results = QuantEvaluator.quantile_analysis(self.factor, self.returns, n_quantiles=5)
        
        self.assertIsInstance(results, dict)
        self.assertIn('Q1_mean_return', results)
        self.assertIn('Q5_mean_return', results)
        self.assertIn('long_short_return', results)


class TestQuantBacktester(unittest.TestCase):
    """Test QuantBacktester class"""
    
    def setUp(self):
        """Set up test data"""
        self.feature_cols = ['book_imbalance', 'price_momentum', 'vwap_deviation', 'spread_feature']
        self.target_col = 'future_returns'
        self.timestamp_col = 'recvTimestamp'
        
        self.pandas_data = TestDataGenerator.create_test_data_pandas(500)
        self.pykx_data = TestDataGenerator.create_test_data_pykx(500)
        # self.numpy_data = TestDataGenerator.create_test_data_numpy(500)
        
        self.backtester = QuantBacktester(
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            timestamp_col=self.timestamp_col,
            walkahead='5s',
            model_type='linear',
            test_size=0.3
        )
    
    def test_init(self):
        """Test backtester initialization"""
        self.assertEqual(self.backtester.feature_cols, self.feature_cols)
        self.assertEqual(self.backtester.target_col, self.target_col)
        self.assertEqual(self.backtester.walkahead, '5s')
        self.assertEqual(self.backtester.model_type, 'linear')
    
    def test_prepare_data_pandas(self):
        """Test data preparation for pandas"""
        combined, y, timestamps, feature_names = self.backtester._prepare_data(
            self.pandas_data, 'pandas'
        )
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(timestamps, np.ndarray)
        self.assertIsInstance(feature_names, list)
        
        # Check dimensions
        self.assertEqual(combined.shape[1], len(self.feature_cols) + 1)  # features + target
        self.assertEqual(len(feature_names), len(self.feature_cols))
        self.assertEqual(feature_names, self.feature_cols)
    
    def test_prepare_data_pykx(self):
        """Test data preparation for pykx"""
        combined, y, timestamps, feature_names = self.backtester._prepare_data(
            self.pykx_data, 'pykx'
        )
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(timestamps, np.ndarray)
        self.assertIsInstance(feature_names, list)
    
    def test_time_series_split(self):
        """Test time series splitting"""
        combined = np.random.randn(100, 5)  # 4 features + 1 target
        timestamps = np.arange(100)
        
        X_train, X_test, y_train, y_test, ts_train, ts_test = self.backtester._time_series_split(
            combined, timestamps
        )
        
        # Check dimensions
        expected_train_size = int(100 * (1 - self.backtester.test_size))
        expected_test_size = 100 - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)
        
        # Check that features have correct number of columns
        self.assertEqual(X_train.shape[1], 4)  # 4 features
        self.assertEqual(X_test.shape[1], 4)
    
    def test_backtest_pandas(self):
        """Test full backtesting with pandas data"""
        results = self.backtester.backtest(self.pandas_data)
        
        # Check that backtesting completed successfully
        self.assertNotIn('error', results)
        
        # Check basic result structure
        self.assertIn('model_type', results)
        self.assertIn('feature_names', results)
        self.assertIn('n_features', results)
        self.assertIn('train_samples', results)
        self.assertIn('test_samples', results)
        
        # Check that feature names are preserved
        self.assertEqual(results['feature_names'], self.feature_cols)
        
        # Check metrics
        self.assertIn('r2', results)
        self.assertIn('ic', results)
        self.assertIn('rank_ic', results)
        
        # Check that metrics are reasonable
        self.assertIsInstance(results['r2'], float)
        self.assertIsInstance(results['ic'], float)
    
    def test_backtest_pykx(self):
        """Test full backtesting with pykx data"""
        results = self.backtester.backtest(self.pykx_data)
        
        # Check that backtesting completed successfully
        self.assertNotIn('error', results)
        
        # Check basic structure
        self.assertIn('model_type', results)
        self.assertIn('feature_names', results)
        self.assertEqual(results['feature_names'], self.feature_cols)
    
    def test_backtest_with_benchmark_factors(self):
        """Test backtesting with benchmark factors"""
        # Create dummy benchmark factors
        benchmark_factors = {
            'momentum_factor': np.random.randn(150),  # Test sample size
            'mean_reversion_factor': np.random.randn(150)
        }
        
        results = self.backtester.backtest(self.pandas_data, benchmark_factors=benchmark_factors)
        
        # Check that factor correlations are calculated
        self.assertIn('factor_correlations', results)
        self.assertIn('corr_with_momentum_factor', results['factor_correlations'])
        self.assertIn('corr_with_mean_reversion_factor', results['factor_correlations'])
    
    def test_backtest_different_models(self):
        """Test backtesting with different model types"""
        model_types = ['linear', 'lasso', 'ridge']
        
        for model_type in model_types:
            backtester = QuantBacktester(
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                model_type=model_type
            )
            
            results = backtester.backtest(self.pandas_data)
            self.assertNotIn('error', results)
            self.assertEqual(results['model_type'], model_type)
    
    def test_backtest_custom_metrics(self):
        """Test backtesting with custom metrics selection"""
        custom_metrics = ['r2', 'ic', 'hit_rate']
        
        backtester = QuantBacktester(
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            metrics=custom_metrics
        )
        
        results = backtester.backtest(self.pandas_data)
        
        # Check that only requested metrics are present
        for metric in custom_metrics:
            self.assertIn(metric, results)
        
        # Check that unrequested metrics are not present
        self.assertNotIn('max_drawdown', results)
        self.assertNotIn('calmar', results)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test data"""
        self.feature_cols = ['book_imbalance', 'price_momentum']
        self.target_col = 'future_returns'
        self.pandas_data = TestDataGenerator.create_test_data_pandas(200)
    
    def test_quick_backtest(self):
        """Test quick_backtest function"""
        results = quick_backtest(
            data=self.pandas_data,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            walkahead='10s'
        )
        
        self.assertNotIn('error', results)
        self.assertIn('r2', results)
        self.assertEqual(results['walkahead'], '10s')
    
    def test_batch_backtest(self):
        """Test batch_backtest function"""
        datasets = {
            'dataset1': self.pandas_data,
            'dataset2': self.pandas_data.copy()
        }
        
        results = batch_backtest(
            datasets=datasets,
            feature_cols=self.feature_cols,
            target_col=self.target_col
        )
        
        self.assertIn('dataset1', results)
        self.assertIn('dataset2', results)
        
        for dataset_name, result in results.items():
            self.assertNotIn('error', result)
            self.assertIn('r2', result)


if __name__ == '__main__':
    # Set up test environment
    unittest.main(verbosity=2) 