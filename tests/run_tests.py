#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test runner for running all validation scripts
"""

import os
import sys
import argparse
import importlib
from pathlib import Path

# Add project root to Python path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def list_test_modules():
    """List all available test modules in the tests directory"""
    test_dir = Path(__file__).parent
    test_files = []
    
    for file in test_dir.glob("*.py"):
        if file.name.startswith("test_") or file.name.startswith("validate_"):
            if file.name != "__init__.py" and file.name != os.path.basename(__file__):
                test_files.append(file.stem)
    
    return sorted(test_files)


def run_test(test_name):
    """Run a specific test by name"""
    try:
        module = importlib.import_module(f"tests.{test_name}")
        
        # Try to find the main test function
        test_function = None
        
        # Common function names in our test files
        potential_names = [
            f"test_{test_name.replace('test_', '')}", 
            f"validate_{test_name.replace('validate_', '')}", 
            "test_embedding_tie",
            "validate_embedding_tie",
            "validate_moe",
            "test_moe",
            "test_transformer_branch",
            test_name, "main", "run_tests", "test", "validate"
        ]
        
        for potential_name in potential_names:
            if hasattr(module, potential_name):
                test_function = getattr(module, potential_name)
                break
        
        if test_function is None:
            # Look for any function with "test" or "validate" in its name
            for attr_name in dir(module):
                if callable(getattr(module, attr_name)) and ("test" in attr_name or "validate" in attr_name):
                    test_function = getattr(module, attr_name)
                    break
        
        if test_function and callable(test_function):
            print(f"Running test: {test_name}")
            print("=" * 50)
            test_function()
            print("\n")
            return True
        else:
            print(f"Could not find a test function in {test_name}")
            return False
    except Exception as e:
        print(f"Error running test {test_name}: {e}")
        return False


def run_all_tests():
    """Run all available tests"""
    test_modules = list_test_modules()
    print(f"Found {len(test_modules)} test modules")
    
    successful_tests = 0
    failed_tests = 0
    
    for test_module in test_modules:
        print(f"\nRunning test module: {test_module}")
        print("-" * 50)
        success = run_test(test_module)
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
    
    print("=" * 50)
    print(f"Test Results: {successful_tests} successful, {failed_tests} failed")
    print("=" * 50)
    
    return failed_tests == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BiLingualSentimentMPS tests")
    parser.add_argument("--test", type=str, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        tests = list_test_modules()
        print("Available tests:")
        for test in tests:
            print(f"  - {test}")
    elif args.test:
        run_test(args.test)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
