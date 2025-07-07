# sda/utils/ultralytics_xpu_patch.py
import logging
import torch
import os
from pathlib import Path
import sys
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_loader, find_spec

# --- Configuration ---
# Environment variable to enable/disable this patch globally if needed.
# The worker process will set this for the MinerU subprocess.
SDA_ULTRALYTICS_XPU_PATCH_ENABLED_ENV_VAR = "SDA_ULTRALYTICS_XPU_PATCH_ENABLED"

# --- Module-level state for patching ---
_sda_patch_attempted_in_this_process = False
_sda_orig_ultralytics_select_device = None

def _log(level, message, pid_prefix=True):
    """Helper for logging with optional PID."""
    prefix = f"[SDA_XPU_PATCH PID:{os.getpid()}] " if pid_prefix else "[SDA_XPU_PATCH] "
    if level == logging.DEBUG:
        logging.debug(prefix + message)
    elif level == logging.INFO:
        logging.info(prefix + message)
    elif level == logging.WARNING:
        logging.warning(prefix + message)
    elif level == logging.ERROR:
        logging.error(prefix + message)

def _get_original_select_device():
    """Safely gets the original ultralytics select_device function."""
    global _sda_orig_ultralytics_select_device
    if _sda_orig_ultralytics_select_device is not None:
        return _sda_orig_ultralytics_select_device

    try:
        import ultralytics.utils.torch_utils as tu
        if hasattr(tu, 'select_device'):
            # If it's already our wrapper, we need to access the original it stored.
            # This assumes the wrapper stores it as _sda_original_callable.
            if hasattr(tu.select_device, '_sda_original_callable'):
                 _sda_orig_select_device = tu.select_device._sda_original_callable
                 return _sda_orig_select_device
            else: # It's not our wrapper, so it's the original (or some other patch)
                _sda_orig_select_device = tu.select_device
                return _sda_orig_select_device
        _log(logging.WARNING, "ultralytics.utils.torch_utils.select_device not found.")
    except ImportError:
        _log(logging.WARNING, "Ultralytics 'torch_utils' module not found.")
    except Exception as e:
        _log(logging.ERROR, f"Error accessing original select_device: {e}", True)
    return None

def _select_device_xpu_wrapper(device: str = '', verbose: bool = True):
    """XPU-aware wrapper for ultralytics.utils.torch_utils.select_device."""
    original_func = _get_original_select_device()
    requested_device_str = str(device).lower()

    if requested_device_str == 'xpu' or requested_device_str.startswith('xpu:'):
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            try:
                actual_device = torch.device(requested_device_str)
                if verbose:
                    env_selector = os.environ.get('ONEAPI_DEVICE_SELECTOR', 'Not Set')
                    _log(logging.INFO, f"Using XPU device '{actual_device}'. ONEAPI_DEVICE_SELECTOR='{env_selector}'")
                return actual_device
            except RuntimeError as e_runtime:
                _log(logging.ERROR, f"RuntimeError for '{requested_device_str}': {e_runtime}. Falling back to CPU.", True)
                if original_func: return original_func('cpu', verbose)
                return torch.device('cpu')
        else:
            _log(logging.ERROR, f"XPU requested ('{requested_device_str}') but torch.xpu not available/IPEX issue. Falling back to CPU.")
            if original_func: return original_func('cpu', verbose)
            return torch.device('cpu')

    if original_func:
        return original_func(device, verbose)

    _log(logging.WARNING, f"Original select_device is None. Trying torch.device('{device}') directly.")
    try:
        return torch.device(device)
    except RuntimeError as e_runtime_direct:
        _log(logging.ERROR, f"Failed torch.device('{device}'): {e_runtime_direct}. Defaulting to CPU.", True)
        return torch.device('cpu')

def _apply_patch_if_needed():
    """Applies the monkey-patch if conditions are met."""
    global _sda_patch_attempted_in_this_process, _sda_orig_ultralytics_select_device

    if os.environ.get(SDA_ULTRALYTICS_XPU_PATCH_ENABLED_ENV_VAR) != '1':
        _log(logging.DEBUG, "XPU patch not enabled via env var. Skipping patch.")
        return

    if _sda_patch_attempted_in_this_process:
        _log(logging.DEBUG, "Patch already attempted in this process.")
        return
    _sda_patch_attempted_in_this_process = True

    try:
        import ultralytics.utils.torch_utils as tu
        if not hasattr(tu, 'select_device'):
            _log(logging.WARNING, "ultralytics.utils.torch_utils.select_device not found. Cannot apply patch.")
            return

        if hasattr(tu.select_device, '_is_sda_patched_select_device'):
            _log(logging.DEBUG, "select_device already has SDA patch marker.")
            return

        # Backup original function before patching
        _sda_orig_ultralytics_select_device = tu.select_device

        _select_device_xpu_wrapper._is_sda_patched_select_device = True
        _select_device_xpu_wrapper._sda_original_callable = _sda_orig_ultralytics_select_device # Store ref to original
        tu.select_device = _select_device_xpu_wrapper
        _log(logging.INFO, "Successfully applied monkey-patch to ultralytics.utils.torch_utils.select_device.")
    except ImportError:
        _log(logging.WARNING, "Ultralytics torch_utils not found during patch application.")
    except Exception as e:
        _log(logging.ERROR, f"Error applying select_device patch: {e}", True)


class _UltralyticsPatcherImporter(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'ultralytics.utils.torch_utils' or fullname == 'ultralytics.engine.predictor':
            _log(logging.DEBUG, f"Import hook triggered for: {fullname}. Deferring patch application.")
            # The module is about to be loaded. Schedule the patch to run after.
            # Using atexit is a bit of a hack; a more robust way would be to use a custom loader
            # that patches after loading. For now, let's try to patch immediately after this hook returns None.
            # Returning None lets the import proceed normally. Then we try to patch.

            # This hook is primarily to ensure our patching logic runs if this module is imported.
            # The actual patching is done by _apply_patch_if_needed().
            # The critical part is that _apply_patch_if_needed() is called at least once.

            # Let the import proceed, then try to patch.
            # This means _apply_patch_if_needed() called at module level should be sufficient.
            # The hook is more a "signal" that now is a good time if not done yet.
            # However, direct call after this `find_spec` returns might be too soon.
            # The direct call at the end of this patch script is the main mechanism.
            pass # Let the import proceed
        return None # Let other finders handle it

# --- Main patching logic execution ---
# This will run when this patch module is first imported.
# The environment variable check within _apply_patch_if_needed decides if patching occurs.
_apply_patch_if_needed()


# --- Logging to confirm module execution context ---
if __name__ != "__main__": # True when imported
    _log(logging.INFO, f"sda.utils.ultralytics_xpu_patch.py imported/executed. Patch attempted status: {_sda_patch_attempted_in_this_process}", pid_prefix=False)
    startup_path_env = os.environ.get('PYTHONSTARTUP')
    if startup_path_env:
        try:
            if Path(startup_path_env).resolve() == Path(__file__).resolve():
                _log(logging.INFO, f"This script was confirmed as PYTHONSTARTUP (path: {startup_path_env})", pid_prefix=False)
        except Exception as e:
             _log(logging.ERROR, f"Error checking PYTHONSTARTUP path ('{startup_path_env}'): {e}", pid_prefix=False)
else: # True when script is run directly
    logging.basicConfig(level=logging.DEBUG) # More verbose for direct testing
    _log(logging.INFO, "sda.utils.ultralytics_xpu_patch.py executed directly for testing.", pid_prefix=False)
    # The patch would have been attempted by the call above.
    _log(logging.INFO, f"Patch application status on direct run: {_sda_patch_attempted_in_this_process}", pid_prefix=False)
    if _sda_patch_attempted_in_this_process and _sda_orig_ultralytics_select_device:
        print("\nSDA Patch Test calls (requires ultralytics and IPEX):")
        try:
            import ultralytics.utils.torch_utils as tu_test
            print(f"  select_device is patched: {hasattr(tu_test.select_device, '_is_sda_patched_select_device')}")

            print(f"  Testing ('xpu', verbose=True):")
            dev_xpu = tu_test.select_device('xpu', verbose=True)
            print(f"    Returned: {dev_xpu} (type: {type(dev_xpu)})")

            print(f"  Testing ('cpu', verbose=True):")
            dev_cpu = tu_test.select_device('cpu', verbose=True)
            print(f"    Returned: {dev_cpu} (type: {type(dev_cpu)})")

        except Exception as e:
            print(f"  Error during direct test calls: {e}")
    else:
        print("  Patch not effectively applied or original Ultralytics not found; tests skipped.")

# The import hook should be installed if we want to catch later imports,
# but the primary mechanism is the direct call to _apply_patch_if_needed()
# when this module is first imported (e.g., by pdf_parser.py or PYTHONSTARTUP).
# If MinerU uses a separate Python process that imports ultralytics fresh,
# and if PYTHONSTARTUP points to this script, this script will run in that new process.
# The environment variable ULTRALYTICS_XPU_PATCH_ENABLED must be set by the parent (worker) process.

# sys.meta_path.insert(0, _UltralyticsPatcherImporter())
# _log(logging.DEBUG, "Ultralytics import hook installed.", pid_prefix=False)
# The import hook is tricky because it needs to ensure the module it's patching is fully loaded
# before patching, or it needs to be a loader itself. Simpler to rely on PYTHONSTARTUP
# to execute this whole script, which then patches.
