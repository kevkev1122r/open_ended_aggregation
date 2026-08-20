"""Single source of truth for on-disk locations.

Modules live inside the package now, so `os.path.dirname(__file__)` no longer
points at the repo root. Everything that touches data or results goes through
here instead, which is also the one place to change if the data directory moves
(e.g. to a shared drive -- it is ~250 MB and gitignored).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
