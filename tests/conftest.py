"""pytest configuration: add repo root to sys.path so 'import quantforge' works."""
import sys
import os

# Insert the repo root (parent of this tests/ directory) at the front of sys.path.
# This is needed on Windows where the working directory may not be auto-added.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
