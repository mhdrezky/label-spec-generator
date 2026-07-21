"""Label extractor — dual-call pipeline (structure -> content -> measure).

    python main.py [image_path]
    python main.py --all
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from api_client import (
    API_READ_TIMEOUT,
    API_URL,
    MODEL,
    check_api_health,
    model_slug,
    warmup_model,
)
from dual_call.extract import run_dual
from llm_cache import save as save_llm_cache
from llm_cache import set_run_cache_dir
from render_md import render_markdown
from samples import SAMPLE_DIR, sample_paths

IMAGE_FILE = "draft.png"
RESULT_DIR = os.path.join("results", "current")
SPECS_FILENAME = "specs.json"
OUTPUT_MD = "output.md"
EDITOR_LATEST = os.path.join("editor", "latest-specs.json")


def create_result_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"_{model_slug()}_{timestamp}"
    output_dir = os.path.join(RESULT_DIR, folder)
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
    """Persist raw LLM JSON under results/<ts>/llm/ for inspection."""
    llm_dir = os.path.join(output_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    for label, key in (("dual-structure", "structure"), ("dual-content", "content")):
        path = Path(llm_dir) / f"{label}.json"
        if path.is_file():
            continue
        payload = result.get(key)
        if isinstance(payload, dict):
            save_llm_cache(path, label, json.dumps(payload, ensure_ascii=False))


def run_extract(image_path: str, *, skip_api_check: bool = False) -> int:
    """Extract one image. Returns process exit code."""
    started = time.monotonic()

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        return 1

    print(f"Extracting label specs from {image_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print("Pipeline: structure -> content -> measure")
    print(f"Timeout: read={API_READ_TIMEOUT}s, structured_output=json_schema")

    output_dir = create_result_dir()
    set_run_cache_dir(os.path.join(output_dir, "llm"))
    print(f"Output directory: {output_dir}")

    if not skip_api_check and (not check_api_health() or not warmup_model()):
        return 1

    stage_dir = os.path.join(output_dir, "stages")
    os.makedirs(stage_dir, exist_ok=True)

    try:
        result = run_dual(image_path)
    except Exception as exc:
        print(f"Extract failed: {exc}", file=sys.stderr)
        print(f"Wall clock: {_format_elapsed(time.monotonic() - started)}")
        return 1

    save_json(os.path.join(stage_dir, "01_structure.json"), result["structure"])
    save_json(os.path.join(stage_dir, "02_content.json"), result["content"])
    save_json(os.path.join(stage_dir, "03_measure.json"), result["spec"])
    save_llm_snapshots(output_dir, result)
    print(f"Stage snapshots saved to {stage_dir}")
    print(f"LLM snapshots saved to {os.path.join(output_dir, 'llm')}")

    spec = result["spec"]
    specs_path = os.path.join(output_dir, SPECS_FILENAME)
    specs_payload = {
        "source_image": image_path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "extract_method": "dual_call",
        "model": MODEL,
        **{k: v for k, v in spec.items() if k != "warnings"},
        "warnings": spec.get("warnings") or result.get("warnings") or [],
    }
    save_json(specs_path, specs_payload)

    md_path = os.path.join(output_dir, OUTPUT_MD)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(spec, image_path))

    if os.path.isdir(os.path.dirname(EDITOR_LATEST)):
        save_json(EDITOR_LATEST, specs_payload)
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
        print("Error: extract produced no labels.", file=sys.stderr)
        return 2
    return 0


def run_all_samples() -> int:
    """Run extract on every image in sample-data/."""
    paths = sample_paths(Path.cwd())
    if not paths:
        print(f"Error: no images found in {SAMPLE_DIR}/", file=sys.stderr)
        return 1

    if not check_api_health() or not warmup_model():
        return 1

    failed: list[str] = []
    for i, path in enumerate(paths, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(paths)}] {path}")
        print("=" * 60)
        if run_extract(str(path), skip_api_check=True) != 0:
            failed.append(path.name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Done with failures ({len(failed)}/{len(paths)}): {', '.join(failed)}")
        return 1
    print(f"Done — all {len(paths)} samples finished.")
    return 0


def main() -> None:
    argv = sys.argv[1:]
    run_all = "--all" in argv
    image_args = [a for a in argv if not a.startswith("--")]

    if run_all:
        sys.exit(run_all_samples())

    image_path = image_args[0] if image_args else IMAGE_FILE
    sys.exit(run_extract(image_path))


if __name__ == "__main__":
    main()
