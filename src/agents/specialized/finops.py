"""
DEPRECATED - Use src/agents/finops.py instead
This file maintained for backwards compatibility
"""

# Re-export from new location
from ..finops import AdvancedFinOpsAgent as FinOpsAgent

__all__ = ["FinOpsAgent"]