"""Print one plate's lines from reprocessed cache."""

import json
import sys
from pathlib import Path

from dual_call.postprocess import merge_to_spec


def main() -> None:
    run_id = sys.argv[1]
    pids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [5]
    base = Path("results-dual") / run_id
    structure = json.loads((base / "stages/01_structure.json").read_text(encoding="utf-8"))
    content = json.loads((base / "stages/02_content.json").read_text(encoding="utf-8"))
    specs = json.loads((base / "specs.json").read_text(encoding="utf-8"))
    warnings: list[str] = []
    spec = merge_to_spec(structure, content, specs["image_px"], warnings)
    for pid in pids:
        lbl = next(l for l in spec["labels"] if l["label_number"] == pid)
        print(f"plate #{pid} {lbl['width_mm']}x{lbl['height_mm']}")
        for ln in lbl["lines"]:
            print(f"  {ln['text']:8} x={ln['x_mm']} y={ln['y_mm']}")
        print()


if __name__ == "__main__":
    main()
