"""Label extractor pipeline orchestrator (designer-style).

    python main.py [image_path]

Sheet nodes: survey → dimensions → decompose
Per-plate nodes: transcribe → position → size
Deterministic: measure (px→mm)
QC: sheet-level review with optional one retry
"""

import json
import os
import sys
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from api_client import (
    API_READ_TIMEOUT,
    API_URL,
    MODEL,
    check_api_health,
    warmup_model,
)
from llm_cache import set_run_cache_dir
from pipeline import run_pipeline
from render_md import render_markdown

IMAGE_FILE = "draft.png"
RESULT_DIR = "results"
SPECS_FILENAME = "specs.json"
OUTPUT_MD = "output.md"
EDITOR_LATEST = os.path.join("editor", "latest-specs.json")


def create_result_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_label_summary(label: dict) -> str:
    num = label.get("label_number", "?")
    w = label.get("width_mm")
    h = label.get("height_mm")
    dims = f"{w}x{h}mm" if w is not None and h is not None else "dims unknown"
    line_count = len(label.get("lines") or [])
    qty = label.get("quantity", 1)
    first = (label.get("lines") or [{}])[0].get("text", "")
    return f"#{num} ({dims}, qty {qty}): {line_count} lines — {first!r}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    image_path = args[0] if args else IMAGE_FILE

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting label specs from {image_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print("Pipeline: designer (survey → dimensions → decompose → per-plate → measure → QC)")
    print(f"Timeout: read={API_READ_TIMEOUT}s, structured_output=json_schema")

    output_dir = create_result_dir()
    set_run_cache_dir(os.path.join(output_dir, "llm"))
    print(f"Output directory: {output_dir}")

    if not check_api_health() or not warmup_model():
        sys.exit(1)

    try:
        ctx = run_pipeline(image_path, stage_dir=os.path.join(output_dir, "stages"))
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Stage snapshots saved to {os.path.join(output_dir, 'stages')}")

    spec = ctx.to_spec_dict()
    specs_path = os.path.join(output_dir, SPECS_FILENAME)
    save_json(
        specs_path,
        {
            "source_image": image_path,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            **spec,
        },
    )

    md_path = os.path.join(output_dir, OUTPUT_MD)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(spec, image_path))

    if os.path.isdir(os.path.dirname(EDITOR_LATEST)):
        save_json(
            EDITOR_LATEST,
            {
                "source_image": image_path,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                **spec,
            },
        )
        print(f"Editor preview copy saved to {EDITOR_LATEST}")

    print(f"Specs saved to {specs_path}")
    print(f"Markdown summary saved to {md_path}")
    print(f"Total labels: {spec.get('total_labels', 0)}")
    for label in spec.get("labels") or []:
        print(f"  - {format_label_summary(label)}")

    warnings = spec.get("warnings") or []
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  ! {warning}")

    if not spec.get("labels"):
        print("Error: pipeline produced no labels — specs are not usable in the editor.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
