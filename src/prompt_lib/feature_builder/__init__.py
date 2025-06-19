"""
Feature Builder prompt templates.

Contains all prompt templates for the FeatureBuilder processor.
"""

from .factor_build_dev import FACTOR_BUILD_DEV_TEMPLATE
from .factor_build_lead import FACTOR_BUILD_LEAD_TEMPLATE
from .factor_build_eval import FACTOR_BUILD_EVAL_TEMPLATE

__all__ = [
    'FACTOR_BUILD_DEV_TEMPLATE',
    'FACTOR_BUILD_LEAD_TEMPLATE',
    'FACTOR_BUILD_EVAL_TEMPLATE'
] 