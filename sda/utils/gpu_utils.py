import logging
from typing import List, Dict, Any

def get_available_xpu_details() -> List[Dict[str, Any]]:
    """
    Checks for Intel XPU availability and returns details for each available XPU device.
    Each detail is a dictionary, e.g., {"id_str": "level_zero:0", "name": "Intel Arc A770"}.
    Returns an empty list if XPUs are not available or an error occurs.
    """
    xpu_devices: List[Dict[str, Any]] = []
    try:
        import torch
        if not torch.xpu.is_available():
            logging.info("Intel XPU devices are not available (torch.xpu.is_available() is False).")
            return xpu_devices

        # intel_extension_for_pytorch might be needed for full functionality or properties
        try:
            import intel_extension_for_pytorch as ipex # noqa
            logging.info(f"Intel Extension for PyTorch (IPEX) version: {ipex.__version__}")
        except ImportError:
            logging.warning("Intel Extension for PyTorch (IPEX) not found. XPU functionality might be limited.")
            # Depending on strictness, could return empty list here if IPEX is mandatory

        device_count = torch.xpu.device_count()
        logging.info(f"Found {device_count} Intel XPU device(s).")

        for i in range(device_count):
            try:
                device_props = torch.xpu.get_device_properties(i)
                device_info = {
                    "id_str": f"level_zero:{i}", # Format for ONEAPI_DEVICE_SELECTOR
                    "name": device_props.name,
                    "major": device_props.major,
                    "minor": device_props.minor,
                    "total_memory_gb": round(device_props.total_memory / (1024**3), 2)
                }
                xpu_devices.append(device_info)
                logging.info(f"  - Device {i}: {device_props.name}, Mem: {device_info['total_memory_gb']}GB")
            except Exception as e_prop:
                logging.error(f"Could not get properties for XPU device {i}: {e_prop}")
                # Add a placeholder or skip this device
                xpu_devices.append({
                    "id_str": f"level_zero:{i}",
                    "name": f"Intel XPU {i} (properties error)",
                    "error": str(e_prop)
                })

    except ImportError:
        logging.info("PyTorch or torch.xpu is not available. Intel XPU support disabled.")
    except Exception as e:
        logging.error(f"An error occurred while checking for XPU devices: {e}", exc_info=True)

    return xpu_devices

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    available_xpus = get_available_xpu_details()
    if available_xpus:
        print(f"\nAvailable Intel XPU devices for ONEAPI_DEVICE_SELECTOR:")
        for dev in available_xpus:
            print(f"  ID String: {dev['id_str']}, Name: {dev.get('name', 'N/A')}")
    else:
        print("\nNo Intel XPU devices found or PyTorch XPU support not available.")
