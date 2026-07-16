"""Dual-call label extractor (structure -> content).

    python second-main.py [image_path]

Two LLM calls: structure then content on the full sheet.
Deterministic: measure (px->mm) via dual_call.postprocess.merge_to_spec.
"""

import json
import os
import sys
import time
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
from pathlib import Path

from dual_call.extract import run_dual
from llm_cache import save as save_llm_cache
from llm_cache import set_run_cache_dir
from render_md import render_markdown

IMAGE_FILE = "draft.png"
RESULT_DIR = "results-dual"
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
    return f"#{num} ({dims}, qty {qty}): {line_count} lines - {first!r}"


def _format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s ({seconds:.1f}s)"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s ({seconds:.1f}s)"


def save_llm_snapshots(output_dir: str, result: dict) -> None:
    """Always persist raw LLM JSON under results/<ts>/llm/ for inspection."""
    llm_dir = os.path.join(output_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    for label, key in (("dual-structure", "structure"), ("dual-content", "content")):
        path = Path(llm_dir) / f"{label}.json"
        if path.is_file():
            continue
        payload = result.get(key)
        if isinstance(payload, dict):
            save_llm_cache(path, label, json.dumps(payload, ensure_ascii=False))


def main() -> None:
    started = time.monotonic()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    image_path = args[0] if args else IMAGE_FILE

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting label specs (dual-call) from {image_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print("Pipeline: dual (structure -> content -> measure)")
    print(f"Timeout: read={API_READ_TIMEOUT}s, structured_output=json_schema")

    output_dir = create_result_dir()
    set_run_cache_dir(os.path.join(output_dir, "llm"))
    print(f"Output directory: {output_dir}")

    if not check_api_health() or not warmup_model():
        sys.exit(1)

    stage_dir = os.path.join(output_dir, "stages")
    os.makedirs(stage_dir, exist_ok=True)

    try:
        result = run_dual(image_path)
    except Exception as exc:
        print(f"Dual extract failed: {exc}", file=sys.stderr)
        print(f"Wall clock: {_format_elapsed(time.monotonic() - started)}")
        sys.exit(1)

    save_json(os.path.join(stage_dir, "01_structure.json"), result["structure"])
    save_json(os.path.join(stage_dir, "02_content.json"), result["content"])
    save_json(os.path.join(stage_dir, "03_measure.json"), result["spec"])
    save_llm_snapshots(output_dir, result)
    print(f"Stage snapshots saved to {stage_dir}")
    print(f"LLM snapshots saved to {os.path.join(output_dir, 'llm')}")

    spec = result["spec"]
    specs_path = os.path.join(output_dir, SPECS_FILENAME)
    save_json(
        specs_path,
        {
            "source_image": image_path,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "extract_method": "dual_call",
            **{k: v for k, v in spec.items() if k != "warnings"},
            "warnings": spec.get("warnings") or result.get("warnings") or [],
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
                "extract_method": "dual_call",
                **{k: v for k, v in spec.items() if k != "warnings"},
                "warnings": spec.get("warnings") or result.get("warnings") or [],
            },
        )
        print(f"Editor preview copy saved to {EDITOR_LATEST}")

    print(f"Specs saved to {specs_path}")
    print(f"Markdown summary saved to {md_path}")
    print(f"Total labels: {spec.get('total_labels', 0)}")
    for label in spec.get("labels") or []:
        print(f"  - {format_label_summary(label)}")

    warnings = spec.get("warnings") or result.get("warnings") or []
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  ! {warning}")

    print(f"Wall clock: {_format_elapsed(time.monotonic() - started)}")

    if not spec.get("labels"):
        print(
            "Error: dual extract produced no labels - specs are not usable in the editor.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
