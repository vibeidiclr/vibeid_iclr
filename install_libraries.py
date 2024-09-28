import subprocess
import sys

def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[import_name] = __import__(import_name)

required_packages = {
    "argparse": "argparse",
    "os": "os",
    "kaggle": "kaggle",
    "opencv-python": "cv2",
    "subprocess": "subprocess",
    "torch": "torch",
    "torch.nn": "torch.nn",
    "torch.optim": "torch.optim",
    "torch.utils.data": "torch.utils.data",
    "torchvision": "torchvision"
}

for package, import_name in required_packages.items():
    install_and_import(package, import_name)

# Additional imports for submodules
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

print("Libraries installed successfully.")
