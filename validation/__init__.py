import os
import sys
import importlib.util

# Load the root validation.py module dynamically to resolve name clash
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_val_path = os.path.join(root_dir, "validation.py")

if os.path.exists(root_val_path):
    spec = importlib.util.spec_from_file_location("validation_root", root_val_path)
    if spec is not None and spec.loader is not None:
        validation_root = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation_root)
        
        # Expose the symbols from the root validation.py
        for key, val in validation_root.__dict__.items():
            if not key.startswith('_'):
                globals()[key] = val
