"""
Run All Tests
=============
Execute all test modules for Blaze2Cap.

Usage:
    cd Blaze2Cap_full
    python -m test.run_all
"""

import os
import sys
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_test(module_name):
    """Run a test module and return success status."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {module_name}")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("=" * 70)
    print("BLAZE2CAP TEST SUITE")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # List of test modules
    tests = [
        "test.test_model",
        "test.test_loss",
        # "test.test_dataloader",  # Requires dataset - skip by default
    ]
    
    results = {}
    
    for test in tests:
        try:
            success = run_test(test)
            results[test] = "PASS" if success else "FAIL"
        except Exception as e:
            results[test] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {test}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
