"""
Personal Knowledge Summary System
A hybrid reranking and interpretable generation-based personal knowledge auto-summary system.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "Personal Knowledge Auto-Summary System with Hybrid Reranking"

from .config import Config
from .logger import setup_logger

__all__ = ["Config", "setup_logger"]
