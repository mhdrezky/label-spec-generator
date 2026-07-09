"""Label extractor pipeline orchestrator.

    python main.py [image_path]            # hybrid (default)
    python main.py [image_path] --single   # legacy single vision call
    python main.py [image_path] --layered  # full layered (detect stage too)

Hybrid (direction B): extract.py does the proven single-call decomposition +
text, and postprocess.py/calibrate is the geometry baseline. Only plates whose
geometry is flagged impossible get re-measured by the per-crop layered stages
(layered.refine_with_layers) — everything simple keeps the proven path.
"""

import json
import os
import sys
from datetime import datetime

# Windows consoles default to cp1252 and choke on em-dash / arrows in output.
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
from postprocess import postprocess
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


def run_extraction(image_path: str, mode: str) -> tuple[dict, list[str]]:
    """Return (raw_spec, warnings) for the chosen pipeline mode.

    hybrid/single both extract first; layered runs the full staged pipeline.
    The hybrid layered-refinement happens later, inside postprocess's refiner
    hook (only for flagged plates)."""
    warnings: list[str] = []
    if mode == "layered":
        from layered import run_layered

        return run_layered(image_path, warnings), warnings

    from extract import extract_specs

    raw_spec, schema_errors = extract_specs(image_path)
    for error in schema_errors:
        print(f"Schema warning: {error}", file=sys.stderr)
    return raw_spec, warnings


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = "single" if "--single" in sys.argv else "layered" if "--layered" in sys.argv else "hybrid"
    image_path = args[0] if args else IMAGE_FILE

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    labels_desc = {
        "single": "single vision call",
        "layered": "layered (detect->size->position->review)",
        "hybrid": "hybrid (extract + calibrate baseline, layered refine on flagged plates)",
    }
    print(f"Extracting label specs from {image_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Pipeline: {labels_desc[mode]}")
    print(f"Timeout: read={API_READ_TIMEOUT}s, structured_output=json_schema")

    output_dir = create_result_dir()
    print(f"Output directory: {output_dir}")

    if not check_api_health() or not warmup_model():
        sys.exit(1)

    try:
        raw_spec, stage_warnings = run_extraction(image_path, mode)
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    specs_path = os.path.join(output_dir, SPECS_FILENAME)

    if raw_spec.get("error") == "parse_failed":
        save_json(specs_path, {"source_image": image_path, **raw_spec})
        print(f"Error: response is not valid JSON. Saved to {specs_path}", file=sys.stderr)
        sys.exit(1)

    refiner = None
    if mode == "hybrid":
        def refiner(spec_ref: dict, warnings: list[str]) -> None:
            from layered import refine_with_layers

            n = refine_with_layers(spec_ref, image_path, warnings)
            if n:
                print(f"Refined {n} flagged plate(s) via per-crop layered geometry.")

    spec = postprocess(raw_spec, refiner=refiner)
    spec["warnings"] = stage_warnings + spec.get("warnings", [])

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

    # copy for the editor preview, which auto-loads this file on open
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


if __name__ == "__main__":
    main()
