"""
src/__init__.py — Makes src/ a Python Package
===============================================
This file tells Python that src/ is a package.
Without it, imports like `from src.config import X` would fail.
"""

from src.data_loader import DataLoader

__version__ = "1.0.0"
__author__  = "Khalida Khaldi"