# sda/utils/ultralytics_xpu_patch.py
import logging
import torch
import os
from pathlib import Path # For PYTHONSTARTUP check

# Module-level flag to ensure the patch is attempted only once per process
_patch_attempted_in_this_process = False

def _do_patch_ultralytics_select_device():
    """
    Attempts to monkey-patch ultralytics.utils.torch_utils.select_device
    for XPU support.
    """
    global _patch_attempted_in_this_process
    if _patch_attempted_in_this_process:
        logging.debug("SDA Patch: Ultralytics select_device patch already attempted in this process.")
        return

    _patch_attempted_in_this_process = True # Mark that we've entered this function

    _orig_select_device_func = None

    try:
        import ultralytics.utils.torch_utils as tu
        if not hasattr(tu, 'select_device'):
            logging.warning("SDA Patch: ultralytics.utils.torch_utils.select_device not found. Cannot patch.")
            return

        if hasattr(tu.select_device, '_is_sda_patched_select_device'):
            logging.debug("SDA Patch: select_device found to be already patched by SDA (has _is_sda_patched_select_device attr).")
            return # Already patched by this specific logic

        _orig_select_device_func = tu.select_device # Backup the original function

        def select_device_xpu_wrapper(device: str = '', verbose: bool = True):
            # This wrapper uses _orig_select_device_func from the enclosing scope
            requested_device_str = str(device).lower()

            if requested_device_str == 'xpu' or requested_device_str.startswith('xpu:'):
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    try:
                        # PyTorch's torch.device() handles "xpu" and "xpu:N" correctly if IPEX is installed.
                        actual_device = torch.device(requested_device_str)
                        if verbose:
                            env_selector = os.environ.get('ONEAPI_DEVICE_SELECTOR', 'Not Set')
                            logging.info(f"SDA Patched select_device: Using XPU device '{actual_device}'. ONEAPI_DEVICE_SELECTOR='{env_selector}'")
                        return actual_device
                    except RuntimeError as e_runtime:
                        logging.error(f"SDA Patched select_device: RuntimeError creating torch.device for '{requested_device_str}': {e_runtime}. Falling back to CPU.", exc_info=verbose)
                        if _orig_select_device_func:
                            return _orig_select_device_func('cpu', verbose)
                        return torch.device('cpu') # Absolute fallback
                else:
                    logging.error(f"SDA Patched select_device: XPU requested ('{requested_device_str}') but torch.xpu not available or IPEX not installed/configured correctly. Falling back to CPU.")
                    if _orig_select_device_func:
                        return _orig_select_device_func('cpu', verbose)
                    return torch.device('cpu') # Absolute fallback

            # Fallback to Ultralytics's original logic for other devices
            if _orig_select_device_func:
                return _orig_select_device_func(device, verbose)

            # Fallback if _orig_select_device_func is somehow None
            logging.warning(f"SDA Patched select_device: Original select_device is None (should not happen if initial import worked). Trying torch.device('{device}') directly.")
            try:
                return torch.device(device)
            except RuntimeError as e_runtime_direct:
                logging.error(f"SDA Patched select_device: Failed torch.device('{device}'): {e_runtime_direct}. Defaulting to CPU.", exc_info=True)
                return torch.device('cpu')

        select_device_xpu_wrapper._is_sda_patched_select_device = True # Mark our wrapper
        tu.select_device = select_device_xpu_wrapper
        logging.info("SDA: Successfully applied monkey-patch to ultralytics.utils.torch_utils.select_device for XPU support.")

    except ImportError:
        logging.warning("SDA Patch: Ultralytics 'torch_utils' module not found during patch attempt.")
    except Exception as e:
        logging.error(f"SDA Patch: Unexpected error during select_device patching: {e}", exc_info=True)

# --- Apply the patch immediately upon module import ---
# This ensures that if this script is run by PYTHONSTARTUP, the patch is applied.
_do_patch_ultralytics_select_device()

# --- Logging to confirm module execution context ---
if __name__ != "__main__": # True when imported
    # This log will appear if the file is imported by pdf_parser.py OR if run by PYTHONSTARTUP
    logging.info(f"SDA: sda.utils.ultralytics_xpu_patch.py finished execution (imported or via PYTHONSTARTUP). Patch attempted status: {_patch_attempted_in_this_process}")

    # Specifically check if PYTHONSTARTUP was the reason for execution
    startup_path_env = os.environ.get('PYTHONSTARTUP')
    if startup_path_env:
        try:
            # Resolve both paths for a more robust comparison, handling potential symlinks or case differences
            if Path(startup_path_env).resolve() == Path(__file__).resolve():
                logging.info(f"SDA: This script (ultralytics_xpu_patch.py) was confirmed as being executed via PYTHONSTARTUP env var pointing to: {startup_path_env}")
        except Exception as e:
            logging.error(f"SDA: Error checking PYTHONSTARTUP path ('{startup_path_env}'): {e}")

elif __name__ == "__main__": # True when script is run directly (e.g., python path/to/this_script.py)
    # Basic logging setup for direct execution testing
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more verbose output during direct test
    logging.info("SDA: sda.utils.ultralytics_xpu_patch.py executed directly (e.g., for testing).")
    logging.info(f"SDA Patch application status on direct run: {_patch_attempted_in_this_process}")

    if _patch_attempted_in_this_process:
        print("\nSDA Patch Test calls (requires ultralytics and IPEX to be installed in current env):")
        try:
            import ultralytics.utils.torch_utils as tu_test # Re-import to ensure we get the patched version if applicable

            print(f"  Is tu_test.select_device our wrapper? {hasattr(tu_test.select_device, '_is_sda_patched_select_device')}")

            print(f"  Testing patched select_device('xpu', verbose=True):")
            dev_xpu = tu_test.select_device('xpu', verbose=True)
            print(f"    Returned: {dev_xpu} (type: {type(dev_xpu)})")

            print(f"  Testing patched select_device('cpu', verbose=True):")
            dev_cpu = tu_test.select_device('cpu', verbose=True)
            print(f"    Returned: {dev_cpu} (type: {type(dev_cpu)})")

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                 print(f"  Testing patched select_device('cuda:0', verbose=True):")
                 dev_cuda = tu_test.select_device('cuda:0', verbose=True)
                 print(f"    Returned: {dev_cuda} (type: {type(dev_cuda)})")
            else:
                print("  CUDA not available, skipping direct test of select_device('cuda:0').")
        except Exception as e:
            print(f"  Error during direct test calls: {e}")
            logging.error("Error during direct test calls of patched function.", exc_info=True)
    else:
        print("  SDA Patches not applied or original Ultralytics components not found; cannot run direct test calls.")
