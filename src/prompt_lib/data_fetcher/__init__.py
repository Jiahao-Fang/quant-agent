"""
Data Fetcher prompt templates.

Contains all prompt templates for the DataFetcher processor.
"""

from .data_fetcher_lead import DATA_FETCHER_LEAD_TEMPLATE
from .data_fetcher_dev import DATA_FETCHER_DEV_TEMPLATE
from .data_build_debug_1 import DATA_BUILD_DEBUG_1_TEMPLATE
from .data_build_debug_2 import DATA_BUILD_DEBUG_2_TEMPLATE

__all__ = [
    'DATA_FETCHER_LEAD_TEMPLATE',
    'DATA_FETCHER_DEV_TEMPLATE', 
    'DATA_BUILD_DEBUG_1_TEMPLATE',
    'DATA_BUILD_DEBUG_2_TEMPLATE'
] 