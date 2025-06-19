"""
Factor Augmenter prompt templates.

Contains all prompt templates for the FactorAugmenter processor.
"""

from .enhancement_strategy import FACTOR_ENHANCEMENT_STRATEGY_TEMPLATE
from .factor_evaluation import FACTOR_EVALUATION_TEMPLATE

__all__ = [
    'FACTOR_ENHANCEMENT_STRATEGY_TEMPLATE',
    'FACTOR_EVALUATION_TEMPLATE'
] 