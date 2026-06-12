import sys
from pathlib import Path

# helpers/ are standalone same-directory scripts, not a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))
