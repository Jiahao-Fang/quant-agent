"""
Processors module for AI Quant Agent.

This module contains all processor implementations using the new simplified architecture:
- Capability-based decorators for marking processor abilities
- Minimal BaseProcessor with conditional abstract methods
- Dynamic LangGraph subgraph generation
- Full integration with LangGraph built-in features
"""

from .data_fetcher import DataFetcher
from .feature_builder import FeatureBuilder
from .factor_augmenter import FactorAugmenter
from .backtest_runner import BacktestRunner

__all__ = [
    "DataFetcher",
    "FeatureBuilder", 
    "FactorAugmenter",
    "BacktestRunner"
] 