#!/usr/bin/env python3
"""Simple test script to verify the environment is working."""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import pytest
    print(f"pytest version: {pytest.__version__}")
except ImportError as e:
    print(f"pytest import error: {e}")

try:
    import langchain
    print(f"langchain imported successfully")
except ImportError as e:
    print(f"langchain import error: {e}")

print("Test script completed successfully")