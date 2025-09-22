#!/usr/bin/env python3
"""
Code quality checker script for pharmaceutical RAG system.

Runs essential code quality checks locally before commits.
Usage: python scripts/quality_check.py [--fix]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], fix_mode: bool = False) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, str(e)


def check_formatting(fix_mode: bool = False) -> bool:
    """Check code formatting with black."""
    print("🔍 Checking code formatting with black...")

    cmd = ["black"]
    if not fix_mode:
        cmd.extend(["--check", "--diff"])
    cmd.extend(["src", "tests", "scripts", "examples"])

    exit_code, output = run_command(cmd, fix_mode)

    if exit_code == 0:
        print("✅ Code formatting is correct")
        return True
    else:
        print("❌ Code formatting issues found:")
        print(output)
        if fix_mode:
            print("🔧 Formatting has been fixed")
            return True
        return False


def check_imports(fix_mode: bool = False) -> bool:
    """Check import sorting with isort."""
    print("🔍 Checking import sorting with isort...")

    cmd = ["isort"]
    if not fix_mode:
        cmd.extend(["--check-only", "--diff"])
    cmd.extend(["src", "tests", "scripts", "examples"])

    exit_code, output = run_command(cmd, fix_mode)

    if exit_code == 0:
        print("✅ Import sorting is correct")
        return True
    else:
        print("❌ Import sorting issues found:")
        print(output)
        if fix_mode:
            print("🔧 Imports have been sorted")
            return True
        return False


def check_linting() -> bool:
    """Check code linting with flake8."""
    print("🔍 Checking code linting with flake8...")

    cmd = ["flake8", "src", "tests", "scripts", "examples"]
    exit_code, output = run_command(cmd)

    if exit_code == 0:
        print("✅ No linting issues found")
        return True
    else:
        print("❌ Linting issues found:")
        print(output)
        return False


def check_security() -> bool:
    """Check security with bandit."""
    print("🔍 Checking security with bandit...")

    cmd = ["bandit", "-r", "src", "-f", "txt"]
    exit_code, output = run_command(cmd)

    if exit_code == 0:
        print("✅ No security issues found")
        return True
    else:
        print("⚠️ Security scan completed with warnings:")
        print(output)
        return True  # Don't fail on warnings, just inform


def check_dependencies() -> bool:
    """Check dependency vulnerabilities with safety."""
    print("🔍 Checking dependency vulnerabilities with safety...")

    cmd = ["safety", "check"]
    exit_code, output = run_command(cmd)

    if exit_code == 0:
        print("✅ No dependency vulnerabilities found")
        return True
    else:
        print("⚠️ Dependency vulnerabilities found:")
        print(output)
        return True  # Don't fail CI on this, just inform


def run_tests() -> bool:
    """Run unit tests."""
    print("🔍 Running unit tests...")

    cmd = ["pytest", "tests", "-k", "not integration", "-q"]
    exit_code, output = run_command(cmd)

    if exit_code == 0:
        print("✅ All tests passed")
        return True
    else:
        print("❌ Tests failed:")
        print(output)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--fix", action="store_true", help="Fix formatting and import issues automatically"
    )
    parser.add_argument(
        "--no-tests", action="store_true", help="Skip running tests"
    )
    args = parser.parse_args()

    print("🧹 Running code quality checks for pharmaceutical RAG system...")
    print("=" * 60)

    checks = [
        ("Formatting", lambda: check_formatting(args.fix)),
        ("Import sorting", lambda: check_imports(args.fix)),
        ("Linting", check_linting),
        ("Security", check_security),
        ("Dependencies", check_dependencies),
    ]

    if not args.no_tests:
        checks.append(("Tests", run_tests))

    all_passed = True

    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {name} check: {e}")
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("🎉 All quality checks passed!")
        if args.fix:
            print("💡 Code has been automatically formatted")
        sys.exit(0)
    else:
        print("💥 Some quality checks failed")
        if not args.fix:
            print("💡 Try running with --fix to automatically fix formatting issues")
        sys.exit(1)


if __name__ == "__main__":
    main()