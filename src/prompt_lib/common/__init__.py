"""
Common prompt templates and utilities.

Contains shared prompt templates and utilities used across processors.
"""

from .pykx_operation import PYKX_OPERATION_TEMPLATE
from .data_fields_description import DATA_FIELDS_DESCRIPTION_TEMPLATE
from .system_manager import SYSTEM_MANAGER_TEMPLATE

__all__ = [
    'PYKX_OPERATION_TEMPLATE',
    'DATA_FIELDS_DESCRIPTION_TEMPLATE',
    'SYSTEM_MANAGER_TEMPLATE'
] 