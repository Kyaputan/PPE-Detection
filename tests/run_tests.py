#!/usr/bin/env python3
"""
Test runner for PPE detection system
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_tests():
    """Discover and run all tests"""
    # Discover tests in the tests directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
