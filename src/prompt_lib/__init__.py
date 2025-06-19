"""
Prompt Library for AI Quant Agent.

Organized prompt templates by processor type with centralized management.
"""

from .prompt_manager import PromptManager, PromptTemplate, get_prompt_manager, set_prompt_manager

# Import all processor-specific templates
from . import data_fetcher
from . import feature_builder
from . import factor_augmenter
from . import backtest_runner
from . import common

__all__ = [
    'PromptManager',
    'PromptTemplate', 
    'get_prompt_manager',
    'set_prompt_manager',
    'data_fetcher',
    'feature_builder',
    'factor_augmenter',
    'backtest_runner',
    'common'
] 