#!/usr/bin/env python3
"""
Startup Environment Validation Script

Validates that .env is properly configured with real API keys (not placeholders).
Run this at application startup or as part of setup verification.

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --check-all  # Validate all env vars
"""
import os
import re
import sys
from pathlib import Path
from typing import List
from typing import Tuple

# Placeholder patterns to detect
PLACEHOLDER_PATTERNS = [
    r"your[_\-].*",
    r"example[_\-\.].*",
    r"placeholder",
    r"replace[_\-]?this",
    r"changeme",
    r"xxx+",
    r"<.*>",  # <your_key_here>
]

# Critical environment variables that must be set
REQUIRED_VARS = [
    "NVIDIA_API_KEY",
]

# Optional but recommended variables
RECOMMENDED_VARS = [
    "PUBMED_EMAIL",
    "APIFY_TOKEN",
]


def is_placeholder(value: str) -> bool:
    """Check if value appears to be a placeholder."""
    if not value:
        return True

    value_lower = value.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, value_lower):
            return True
    return False


def validate_environment() -> Tuple[bool, List[str], List[str]]:
    """
    Validate environment configuration.

    Returns:
        Tuple of (success, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        errors.append(
            "❌ .env file not found!\n" "   Run: cp .env.example .env\n" "   Then edit .env with your real API keys"
        )
        return False, errors, warnings

    # Check required variables
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if not value:
            errors.append(f"❌ {var} is not set\n" f"   Add it to your .env file")
        elif is_placeholder(value):
            errors.append(f"❌ {var} appears to be a placeholder: '{value}'\n" f"   Replace with a real API key in .env")
        elif len(value) < 20:
            errors.append(f"❌ {var} is too short (must be at least 20 characters)\n" f"   Current length: {len(value)}")

    # Check recommended variables
    for var in RECOMMENDED_VARS:
        value = os.getenv(var)
        if not value:
            warnings.append(
                f"⚠️  {var} is not set (optional but recommended)\n"
                f"   Add it to your .env file if you need this feature"
            )
        elif is_placeholder(value):
            warnings.append(
                f"⚠️  {var} appears to be a placeholder: '{value}'\n"
                f"   Replace with a real value if you need this feature"
            )

    # Check for common mistakes
    if env_file.read_text().count("your_") > 3:
        warnings.append(
            "⚠️  Multiple placeholder values detected in .env\n" "   Review your .env file and replace all placeholders"
        )

    success = len(errors) == 0
    return success, errors, warnings


def main():
    """Main validation entry point."""
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    success, errors, warnings = validate_environment()

    if errors:
        print("🚨 Environment Validation Failed:\n")
        for error in errors:
            print(error)
            print()

    if warnings:
        print("⚠️  Warnings:\n")
        for warning in warnings:
            print(warning)
            print()

    if success and not warnings:
        print("✅ Environment validation passed!")
        print("   All required configuration is properly set.")
        return 0
    elif success:
        print("✅ Environment validation passed (with warnings)")
        return 0
    else:
        print("\n💡 Quick Fix:")
        print("   1. cp .env.example .env")
        print("   2. Edit .env and replace all placeholders with real values")
        print("   3. Run: python scripts/validate_env.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
