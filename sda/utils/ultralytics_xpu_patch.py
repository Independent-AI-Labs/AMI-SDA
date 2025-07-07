# sda/utils/ultralytics_xpu_patch.py
import logging
import torch
import os

# Flag to ensure patch is applied only once per process
_patch_applied_flag = False
_orig_select_device = None
_orig_predictor_setup_model = None

def apply_ultralytics_xpu_patch():
    global _orig_select_device, _orig_predictor_setup_model, _patch_applied_flag

    if _patch_applied_flag:
        # logging.debug("Ultralytics XPU patch already applied or attempted.") # Becomes noisy if PYTHONSTARTUP
        return

    # --- 1. Patch ultralytics.utils.torch_utils.select_device ---
    try:
        import ultralytics.utils.torch_utils as tu
        if hasattr(tu, 'select_device') and tu.select_device.__name__ != 'select_device_xpu_wrapper':
            _orig_select_device = tu.select_device

            def select_device_xpu_wrapper(device: str = '', verbose: bool = True):
                global _orig_select_device
                requested_device_str = str(device).lower()

                if requested_device_str == 'xpu' or requested_device_str.startswith('xpu:'):
                    if hasattr(torch, 'xpu') and torch.xpu.is_available():
                        try:
                            # For 'xpu' or 'xpu:N', PyTorch can create the device object.
                            # The ONEAPI_DEVICE_SELECTOR in env should guide 'xpu' to the correct tile.
                            actual_device = torch.device(requested_device_str)
                            if verbose:
                                logging.info(f"Patched select_device: Using XPU device '{actual_device}'. ONEAPI_DEVICE_SELECTOR='{os.environ.get('ONEAPI_DEVICE_SELECTOR')}'")
                            return actual_device
                        except RuntimeError as e:
                            logging.error(f"Patched select_device: Error creating torch.device for '{requested_device_str}': {e}. Falling back to CPU.", exc_info=True)
                            return _orig_select_device('cpu', verbose) if _orig_select_device else torch.device('cpu')
                    else:
                        logging.error(f"Patched select_device: XPU requested ('{requested_device_str}') but torch.xpu not available. Falling back to CPU.")
                        return _orig_select_device('cpu', verbose) if _orig_select_device else torch.device('cpu')

                # Fallback to original for non-XPU devices
                if _orig_select_device:
                    return _orig_select_device(device, verbose)
                else: # Should not happen if ultralytics is installed
                    logging.warning(f"Patched select_device: Original select_device not found. Trying torch.device('{device}').")
                    return torch.device(device)

            tu.select_device = select_device_xpu_wrapper
            logging.info("Applied monkey-patch to ultralytics.utils.torch_utils.select_device for XPU support.")
            _patch_applied_flag = True # Mark patch as applied after the first successful one
        elif hasattr(tu, 'select_device') and tu.select_device.__name__ == 'select_device_xpu_wrapper':
            logging.debug("select_device already patched.")
            _patch_applied_flag = True


    except ImportError:
        logging.warning("Ultralytics 'torch_utils' not found. Cannot patch select_device.")
    except AttributeError:
        logging.warning("Ultralytics 'select_device' not found in torch_utils. Cannot patch.")
    except Exception as e:
        logging.error(f"Unexpected error patching select_device: {e}", exc_info=True)

    # --- 2. (Optional but recommended by user) Patch ultralytics.engine.predictor.BasePredictor.setup_model ---
    # This patch is to more forcefully ensure device='xpu' is used if passed,
    # potentially overriding some internal logic in setup_model that might try to default to CUDA.
    try:
        import ultralytics.engine.predictor as pred
        if hasattr(pred, 'BasePredictor') and hasattr(pred.BasePredictor, 'setup_model') and \
           pred.BasePredictor.setup_model.__name__ != 'setup_model_xpu_wrapper':
            _orig_predictor_setup_model = pred.BasePredictor.setup_model

            def setup_model_xpu_wrapper(self, model, verbose=False):
                global _orig_predictor_setup_model

                # If the intended device (from args) is 'xpu', ensure it's passed to the (patched) select_device
                current_arg_device = str(self.args.device).lower()
                if current_arg_device == 'xpu' or current_arg_device.startswith('xpu:'):
                    if verbose:
                        logging.info(f"Patched BasePredictor.setup_model: Ensuring device '{current_arg_device}' is used via (patched) select_device.")
                    # The actual device selection happens inside the original setup_model via select_device,
                    # which should now be our patched version.
                    # No need to force self.device directly here if select_device is patched.

                if _orig_predictor_setup_model:
                    return _orig_predictor_setup_model(self, model, verbose)
                else:
                    # This is a fallback and might not be complete if original is missing
                    logging.warning("Original BasePredictor.setup_model not available for fallback in wrapper.")
                    self.model = model
                    # Use the (potentially patched) select_device from tu
                    self.device = tu.select_device(self.args.device, verbose=verbose)
                    if getattr(self.args, 'half', False): # Safely get 'half'
                        self.model.half()
                    self.model.to(self.device).eval() # Move model to device and eval
                    return self.model

            pred.BasePredictor.setup_model = setup_model_xpu_wrapper
            logging.info("Applied monkey-patch to ultralytics.engine.predictor.BasePredictor.setup_model for XPU support.")
            _patch_applied_flag = True # Mark patch as applied
        elif hasattr(pred, 'BasePredictor') and hasattr(pred.BasePredictor, 'setup_model') and pred.BasePredictor.setup_model.__name__ == 'setup_model_xpu_wrapper':
            logging.debug("BasePredictor.setup_model already patched.")
            _patch_applied_flag = True


    except ImportError:
        logging.warning("Ultralytics 'engine.predictor' not found. Cannot patch BasePredictor.setup_model.")
    except AttributeError:
        logging.warning("Ultralytics 'BasePredictor.setup_model' not found. Cannot patch.")
    except Exception as e:
        logging.error(f"Unexpected error patching BasePredictor.setup_model: {e}", exc_info=True)

    if not _patch_applied_flag:
        logging.warning("No Ultralytics patches were successfully applied.")


# Apply patches when this module is imported.
# This is primarily for when PYTHONSTARTUP points to this file.
apply_all_patches()

# Log to confirm execution via PYTHONSTARTUP
startup_path = os.environ.get('PYTHONSTARTUP')
if startup_path and Path(startup_path).resolve() == Path(__file__).resolve():
    logging.info(f"sda.utils.ultralytics_xpu_patch executed via PYTHONSTARTUP. Patch application status: {_patch_applied_flag}")
elif __name__ != "__main__":
     # Also log if imported normally, though PYTHONSTARTUP is the target for MinerU
    logging.info(f"sda.utils.ultralytics_xpu_patch imported. Patch application status: {_patch_applied_flag}")
