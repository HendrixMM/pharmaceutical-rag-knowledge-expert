"""
Modular medical safety components.

Focused modules for pharmaceutical safety compliance and guardrails.
"""

# Import key functions for backward compatibility
from .disclaimer_management import (
    contains_medical_disclaimer,
    insert_medical_disclaimer,
    format_disclaimer_consistently
)

from .source_metadata_utils import (
    update_source_metadata,
    set_source_flag,
    append_source_warning,
    _get_source_value,
    _clone_source_for_update,
    _ensure_metadata_dict
)

__all__ = [
    # Disclaimer management
    'contains_medical_disclaimer',
    'insert_medical_disclaimer',
    'format_disclaimer_consistently',

    # Source metadata utilities
    'update_source_metadata',
    'set_source_flag',
    'append_source_warning',
    '_get_source_value',
    '_clone_source_for_update',
    '_ensure_metadata_dict',
]