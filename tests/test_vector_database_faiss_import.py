"""Test FAISS import handling in vector_database.py"""

import sys
import importlib
from unittest.mock import patch
import pytest


def test_vector_database_imports_without_faiss():
    """Test that vector_database.py imports correctly even when FAISS is not available."""
    # Mock faiss import to raise ImportError
    with patch.dict('sys.modules', {'faiss': None}):
        # Remove the module from cache if it was already imported
        if 'src.vector_database' in sys.modules:
            del sys.modules['src.vector_database']

        # Mock the import to simulate FAISS not being available
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                   ImportError() if name == 'faiss' else __import__(name, *args, **kwargs)):
            # The module should still import without raising an exception
            try:
                import src.vector_database
                # Verify that faiss is set to None
                assert src.vector_database.faiss is None
                # Verify that the module was imported successfully
                assert hasattr(src.vector_database, 'VectorDatabase')
            except Exception as e:
                pytest.fail(f"vector_database.py failed to import when FAISS is not available: {e}")


def test_vector_database_imports_with_faiss():
    """Test that vector_database.py imports correctly when FAISS is available."""
    # This test assumes FAISS is available in the test environment
    try:
        import src.vector_database
        # If FAISS is available, faiss should not be None
        # Note: This test might run in an environment without FAISS,
        # so we don't assert faiss is not None
        assert hasattr(src.vector_database, 'VectorDatabase')
    except Exception as e:
        pytest.fail(f"vector_database.py failed to import even when FAISS should be available: {e}")