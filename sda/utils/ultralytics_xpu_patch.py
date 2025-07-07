# sda/utils/ultralytics_xpu_patch.py
import logging
import torch
import ultralytics.utils.torch_utils as tu

# It's important to attempt the import and patch only once.
# A simple way to ensure this is by checking if the original function has already been patched.
# We can check if tu.select_device is already our wrapper.

_patch_applied_flag = False

def apply_ultralytics_xpu_patch():
    global _orig_select_device, _patch_applied_flag

    if _patch_applied_flag:
        logging.debug("Ultralytics XPU patch already applied.")
        return

    try:
        # 1. Backup original if it's not already our wrapper
        if hasattr(tu, 'select_device') and tu.select_device.__name__ == 'select_device_xpu':
            logging.debug("Ultralytics XPU patch appears to be already in place (by name check).")
            _patch_applied_flag = True
            return

        _orig_select_device = tu.select_device
        logging.info("Original ultralytics.utils.torch_utils.select_device backed up.")

        # 2. Define XPU-aware wrapper
        def select_device_xpu(device: str = '', verbose: bool = True):
            global _orig_select_device
            requested_device_str = str(device).lower()

            if requested_device_str == 'xpu':
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    logging.info(f"XPU device requested and torch.xpu.is_available() is True. Returning torch.device('xpu').")
                    # Ultralytics's original select_device can return a torch.device object.
                    return torch.device('xpu')
                else:
                    logging.error("Requested device 'xpu' but torch.xpu is not available or IPEX not installed correctly.")
                    # Fallback to CPU if XPU requested but not available, and log a clear warning.
                    # Or, could raise an error: raise RuntimeError("Requested device 'xpu' but no Intel XPU available")
                    logging.warning("Falling back to CPU due to XPU unavailability.")
                    return _orig_select_device('cpu', verbose) # Fallback to CPU

            elif requested_device_str.startswith("xpu:"):
                 if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    try:
                        # Attempt to return a specific XPU device if format is "xpu:N"
                        # This matches how CUDA devices can be specified e.g. "cuda:0"
                        specific_xpu_device = torch.device(requested_device_str)
                        # Check if the specific device index is valid
                        # torch.xpu.device_count() would be useful here if we need to validate N
                        # For now, assume torch.device(requested_device_str) handles validation or errors appropriately
                        logging.info(f"Specific XPU device '{requested_device_str}' requested. Returning torch.device('{requested_device_str}').")
                        return specific_xpu_device
                    except Exception as e:
                        logging.error(f"Error processing specific XPU device string '{requested_device_str}': {e}. Falling back.", exc_info=True)
                        # Fallback to generic XPU or CPU
                        logging.warning(f"Falling back to generic XPU for '{requested_device_str}'.")
                        return torch.device('xpu') # Try generic XPU
                 else:
                    logging.error(f"Requested device '{requested_device_str}' but torch.xpu is not available. Falling back to CPU.")
                    return _orig_select_device('cpu', verbose)


            # Fallback to Ultralytics's original logic for other devices (cpu, cuda:N, mps, etc.)
            if verbose:
                logging.debug(f"XPU not requested ('{device}'). Calling original select_device.")
            return _orig_select_device(device, verbose)

        # 3. Monkey-patch Ultralytics
        tu.select_device = select_device_xpu
        _patch_applied_flag = True
        logging.info("Ultralytics XPU patch applied: ultralytics.utils.torch_utils.select_device is now select_device_xpu.")

    except AttributeError:
        logging.error("Failed to patch Ultralytics: `ultralytics.utils.torch_utils.select_device` not found. Maybe Ultralytics version changed or it's not installed?", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred while applying Ultralytics XPU patch: {e}", exc_info=True)

# Apply the patch when this module is imported.
# Ensure this module is imported early in the application lifecycle.
apply_ultralytics_xpu_patch()
