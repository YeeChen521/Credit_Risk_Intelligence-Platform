import sys
from pathlib import Path

# When uvicorn's reload subprocess imports "src.api.main", Python runs this
# __init__.py before executing main.py.  At that point sys.path contains only
# the project root (injected via PYTHONPATH by the parent process) but NOT
# src/ or src/api/, so bare-name imports like `from db import` would fail.
# Adding both here fixes that for any process that imports the api package.
_api_dir = str(Path(__file__).resolve().parent)          # src/api/
_src_dir = str(Path(__file__).resolve().parent.parent)   # src/

for _p in (_src_dir, _api_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
