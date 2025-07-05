# run_sanity_tests.py

import subprocess
import sys
from pathlib import Path

def main():
    """
    Runs designated test scripts and reports their success or failure.
    Currently, it only runs the sanity import test.
    """
    print("Starting Custom Test Runner...")

    test_script_path = Path("tests") / "test_sanity_imports.py"

    if not test_script_path.exists():
        print(f"ERROR: Test script not found: {test_script_path}", file=sys.stderr)
        print("Custom Test Runner: FAILED")
        sys.exit(1)

    print(f"\nExecuting test script: {test_script_path}...")

    # Ensure we're using the same Python interpreter that's running this script
    python_executable = sys.executable

    process = subprocess.run(
        [python_executable, str(test_script_path)],
        capture_output=True,
        text=True,
        check=False # We check the returncode manually
    )

    print("\n--- Output from test_sanity_imports.py ---")
    if process.stdout:
        print("Stdout:\n" + process.stdout)
    if process.stderr:
        # Prepend [STDERR] to make it distinct in logs
        for line in process.stderr.splitlines():
            print(f"[STDERR] {line}")
    print("--- End of test_sanity_imports.py output ---")

    if process.returncode == 0:
        print("\nSanity Imports: PASSED")
        print("Custom Test Runner: FINISHED SUCCESSFULLY")
        sys.exit(0)
    else:
        print(f"\nSanity Imports: FAILED (return code: {process.returncode})")
        print("Custom Test Runner: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
