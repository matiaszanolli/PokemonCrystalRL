#!/usr/bin/env python3
"""
run_tests.py - Test Runner for Pokemon Crystal RL Agent

This script provides a convenient interface to run the comprehensive test suite
with different configurations and reporting options.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, markers=None, parallel=False):
    """
    Run the test suite with specified parameters
    
    Args:
        test_type: Type of tests to run (all, unit, integration, performance)
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Specific pytest markers to run
        parallel: Enable parallel test execution
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration": 
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "semantic":
        cmd.extend(["-m", "semantic"])
    elif test_type == "dialogue":
        cmd.extend(["-m", "dialogue"])
    elif test_type == "choice":
        cmd.extend(["-m", "choice"])
    elif test_type == "database":
        cmd.extend(["-m", "database"])
    elif markers:
        cmd.extend(["-m", markers])
    
    # Verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.extend(["-q"])
    
    # Coverage
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("⚠️  pytest-xdist not available, running tests sequentially")
    
    # Test directory
    cmd.append("tests/")
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    # Execute tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner interface"""
    parser = argparse.ArgumentParser(description="Run Pokemon Crystal RL Agent tests")
    
    parser.add_argument(
        "--type", "-t", 
        choices=["all", "unit", "integration", "performance", "semantic", "dialogue", "choice", "database"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Run tests with specific markers"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only fast unit tests"
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke tests only (basic functionality)"
    )
    
    args = parser.parse_args()
    
    # Quick test configuration
    if args.quick:
        markers = "unit and not slow"
    elif args.smoke:
        markers = "unit and not slow and not database"
    else:
        markers = args.markers
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=markers,
        parallel=args.parallel
    )
    
    # Print summary
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
