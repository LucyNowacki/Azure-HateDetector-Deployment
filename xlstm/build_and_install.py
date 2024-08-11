import subprocess
import os

# Clear the PyTorch extensions cache
def clear_torch_extensions_cache():
    torch_extensions_path = os.path.expanduser("~/.cache/torch_extensions/")
    if os.path.exists(torch_extensions_path):
        subprocess.run(["rm", "-rf", torch_extensions_path])

# Clear the cache
clear_torch_extensions_cache()

# Set environment variables for GCC 11
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'

# Build the package
subprocess.run(["python", "setup.py", "build_ext", "--inplace"], check=True)

# Install the package
subprocess.run(["pip", "install", "."], check=True)
