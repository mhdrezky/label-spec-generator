"""Run main.py --all (every image in sample-data/).

    uv run python scripts/run_samples.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    return subprocess.run(["uv", "run", "main.py", "--all"], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
