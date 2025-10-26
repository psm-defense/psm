"""
SPE-SC Source Package

This package contains the core functionality for the SPE-SC project,
including defense mechanisms, metrics, and utilities.
"""

# Make key functions and classes available at package level
from .utils import load_dataset_jsonl
from .metrics import rouge_recall

__version__ = "1.0.0"
__all__ = ["load_dataset_jsonl", "rouge_recall"]
