"""Re-run merge_to_spec on cached dual-call stage JSON."""

import json
import sys
from pathlib import Path

from dual_call.postprocess import merge_to_spec


def reprocess(run_id: str) -> tuple[dict, list[str]]:
    base = Path("results") / run_id
    structure = json.loads((base / "stages/01_structure.json").read_text(encoding="utf-8"))
    content = json.loads((base / "stages/02_content.json").read_text(encoding="utf-8"))
    image_px = structure.get("image_px") or content.get("image_px")
    if not image_px:
        specs = json.loads((base / "specs.json").read_text(encoding="utf-8"))
        image_px = specs["image_px"]
    warnings: list[str] = []
    spec = merge_to_spec(structure, content, image_px, warnings)
    return spec, warnings


def main() -> None:
    runs = sys.argv[1:] or ["20260716_091406"]
    for run in runs:
        spec, warnings = reprocess(run)
        print(f"=== {run} ===")
        for lbl in spec.get("labels") or []:
            num = lbl.get("label_number")
            ys = sorted(
                {round(l["y_mm"], 1) for l in lbl["lines"] if l.get("y_mm") is not None}
            )
            xs = [round(l["x_mm"], 1) for l in lbl["lines"][:3]]
            w, h = lbl.get("width_mm"), lbl.get("height_mm")
            print(f"  plate #{num}: {w}x{h} rows y={ys} sample_x={xs}")
        for w in warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
