# tests/test_sanity_imports.py

import importlib
import os
import pkgutil
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing 'sda'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure sda can be imported
try:
    import sda
except ImportError as e:
    print(f"FATAL: Failed to import sda package. PROJECT_ROOT: {PROJECT_ROOT}, sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1) # Critical error, stop

def get_all_modules_in_package(package_name_str: str) -> list[str]:
    """
    Recursively finds all importable module strings within a given package.
    """
    package = importlib.import_module(package_name_str)
    modules = []

    module_paths = package.__path__  # list of paths for the package

    # Add the top-level package itself
    modules.append(package_name_str)

    for _finder, name, _ispkg in pkgutil.walk_packages(path=module_paths, prefix=package.__name__ + '.'):
        modules.append(name)

    return sorted(list(set(modules)))

def main():
    """
    Attempts to import all modules in the 'sda' package and reports results.
    Exits with 0 on full success, 1 on any failure.
    """
    print("Starting Sanity Import Test (standalone script)...")

    try:
        modules_to_test = get_all_modules_in_package("sda")
    except Exception as e:
        print(f"ERROR: Failed during module discovery for 'sda': {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    if not modules_to_test:
        print("WARNING: No modules found to test for 'sda'. Ensure package is not empty and discovery is correct.", file=sys.stderr)
        # Exiting with 0 as technically no *imports* failed, but this is a problematic state.
        # Depending on strictness, could exit 1. For now, let's say an empty list is "no failures".
        sys.exit(0)

    print(f"Discovered {len(modules_to_test)} modules/packages under 'sda' to test for importability.")

    failed_imports = []
    successful_imports_count = 0

    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            # print(f"  OK: {module_name}") # Verbose success
            successful_imports_count += 1
        except Exception as e:
            print(f"FAIL: {module_name} - {type(e).__name__}: {e}", file=sys.stderr)
            failed_imports.append((module_name, e))

    print("\n--- Sanity Import Test Summary ---")
    print(f"Total modules/packages attempted: {len(modules_to_test)}")
    print(f"Successfully imported: {successful_imports_count}")
    print(f"Failed to import: {len(failed_imports)}")

    if failed_imports:
        print("\nDetails of failed imports:")
        for name, err in failed_imports:
            print(f"  - Module: {name}, Error: {type(err).__name__}: {err}", file=sys.stderr)
        print("\nSanity Import Test: FAILED")
        sys.exit(1)

    print("\nSanity Import Test: PASSED - All discovered sda modules imported successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
