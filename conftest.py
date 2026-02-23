"""
conftest.py — pytest Configuration
====================================
This file lives in the project root and tells pytest
to add the project root to Python's path.

Without this, pytest can't find the src/ package and
throws: ModuleNotFoundError: No module named 'src'

This file is automatically loaded by pytest before
any tests run — no imports needed.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))