"""
Backtest Runner prompt templates.

Contains all prompt templates for the BacktestRunner processor.
"""

from .strategy_design import STRATEGY_DESIGN_TEMPLATE
from .performance_analysis import PERFORMANCE_ANALYSIS_TEMPLATE

__all__ = [
    'STRATEGY_DESIGN_TEMPLATE',
    'PERFORMANCE_ANALYSIS_TEMPLATE'
] 