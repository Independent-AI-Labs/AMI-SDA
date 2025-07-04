# Vulkan Support for `sentence-transformers`

Using Vulkan for computation with `sentence-transformers` (and underlying `torch`) is an experimental feature and requires a manual build of PyTorch with the Vulkan backend enabled.

This is a complex process and is not recommended unless you are an advanced user.

## High-Level Steps

1.  **Install Vulkan SDK:** Download and install the Vulkan SDK for your operating system from the official website: [https://vulkan.lunarg.com/](https://vulkan.lunarg.com/)

2.  **Build PyTorch from Source:** You will need to clone the PyTorch repository and build it from source, ensuring that the Vulkan backend is enabled during the build configuration process.

    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # Follow PyTorch build instructions, making sure to enable the Vulkan backend.
    # This usually involves setting environment variables like USE_VULKAN=1
    python setup.py install
    ```
    Windows Example:
    ```powershell
    # Tell CMake to use the VS 2022 generator
    $Env:CMAKE_GENERATOR = "Visual Studio 17 2022"
    
    $Env:USE_VULKAN = 1
    $Env:USE_VULKAN_SHADERC_RUNTIME = 1
    $Env:USE_VULKAN_WRAPPER = 0
    $Env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"
    
    # Now invoke the PyTorch installer
    python setup.py install
    ```

3.  **Install Other Dependencies:** Install the remaining dependencies from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure `config.py`:** Set `EMBEDDING_DEVICE = "vulkan"` in your `config.py` file.

For detailed and up-to-date instructions, please refer to the official PyTorch documentation on building from source.