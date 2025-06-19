"""
pytest configuration file for quant-agent project.

This file automatically configures the Python path so that tests can import
from the src/ directory without issues.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

# Add both project root and src to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Print path info for debugging (remove in production)
print(f"✅ Added to Python path: {project_root}")
print(f"✅ Added to Python path: {src_path}") 