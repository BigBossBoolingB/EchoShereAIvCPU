#!/usr/bin/env python3
"""
Build script for EchoSphere AI-vCPU development workflow.

This script provides a unified interface for common development tasks
including dependency installation, code quality checks, testing, and
documentation generation.
"""

import argparse
import subprocess
import sys


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return False


def install_deps() -> bool:
    """Install development dependencies."""
    return run_command("pip install -e '.[dev]'", "Installing dependencies")


def format_code() -> bool:
    """Format code with Black."""
    return run_command("black .", "Formatting code with Black")


def sort_imports() -> bool:
    """Sort imports with isort."""
    return run_command("isort .", "Sorting imports with isort")


def lint_code() -> bool:
    """Run linting with Flake8."""
    return run_command("flake8 .", "Linting code with Flake8")


def type_check() -> bool:
    """Run type checking with MyPy."""
    return run_command("mypy echosphere/", "Type checking with MyPy")


def run_tests() -> bool:
    """Run tests with pytest."""
    return run_command("pytest", "Running tests")


def run_tests_with_coverage() -> bool:
    """Run tests with coverage reporting."""
    return run_command(
        "pytest --cov=echosphere", "Running tests with coverage"
    )


def run_performance_tests() -> bool:
    """Run performance benchmarks."""
    return run_command(
        "pytest tests/ --durations=10", "Running performance benchmarks"
    )


def quality_check() -> bool:
    """Run all code quality checks."""
    checks = [
        ("Formatting", format_code),
        ("Import sorting", sort_imports),
        ("Linting", lint_code),
        ("Type checking", type_check),
    ]

    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"❌ {name} check failed")

    return all_passed


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="EchoSphere build script")
    parser.add_argument(
        "--install", action="store_true", help="Install dependencies"
    )
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument(
        "--type-check", action="store_true", help="Run type checking"
    )
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument(
        "--test-cov", action="store_true", help="Run tests with coverage"
    )
    parser.add_argument(
        "--perf", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--quality", action="store_true", help="Run all quality checks"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run complete build pipeline"
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    success = True

    if args.install or args.all:
        success &= install_deps()

    if args.format or args.all:
        success &= format_code()
        success &= sort_imports()

    if args.lint or args.quality or args.all:
        success &= lint_code()

    if args.type_check or args.quality or args.all:
        success &= type_check()

    if args.test or args.all:
        success &= run_tests()

    if args.test_cov:
        success &= run_tests_with_coverage()

    if args.perf:
        success &= run_performance_tests()

    if args.quality and not args.all:
        success = quality_check()

    if success:
        print("\n🎉 Build completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
