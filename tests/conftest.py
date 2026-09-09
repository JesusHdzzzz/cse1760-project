import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_dir in (ROOT / "part1" / "src", ROOT / "part2" / "src", ROOT / "part3" / "src"):
    sys.path.insert(0, str(source_dir))
