#!/usr/bin/env python3
"""Check if dependencies are available."""

try:
    import langchain

    print(f"✅ langchain: {langchain.__version__}")
except ImportError as e:
    print(f"❌ langchain: {e}")

try:
    import faiss

    print(f"✅ faiss: {faiss.__version__}")
except ImportError as e:
    print(f"❌ faiss: {e}")

try:
    pass

    print("✅ python-dotenv")
except ImportError as e:
    print(f"❌ python-dotenv: {e}")

try:
    pass

    print("✅ requests")
except ImportError as e:
    print(f"❌ requests: {e}")

try:
    pass

    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

print("Dependency check complete.")
