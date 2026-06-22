import json
import os
import sys
from datetime import datetime
from pathlib import Path

from api_client import (
    API_READ_TIMEOUT,
    API_SPECS_MAX_TOKENS,
    API_URL,
    MODEL,
    API_ENABLE_THINKING,
    call_chat,
    check_api_health,
    parse_json_response,
    warmup_model,
)

RESULT_DIR = "results"
OUTPUT_MD = "output.md"
SPECS_FILENAME = "specs.json"


def build_specs_prompt(markdown: str) -> str:
    return f"""
You are given a label layout analysis in Markdown format.
Convert it into manufacturing label specifications as JSON only. No markdown, no explanation.

## Input analysis

{markdown}

## Output rules

Read the **Plate Inventory** and **Section Definitions** from the analysis carefully.

### Plate counting logic
- `plate_divider` section: each cell in a divided row = one separate label plate
- `layout_guide` section: each horizontal row = one label plate; columns are layout guides, NOT separate plates
- NEVER merge plates from different sections into one label
- NEVER treat vertical columns as one tall plate spanning multiple sections
- Total labels in output MUST match **Total_Physical_Plates** from the analysis

### Per-label dimensions
- `label_length_mm` and `label_height_mm` = that plate's **Plate_Media_Width_mm** and **Plate_Media_Height_mm** from its section
- Do NOT use full canvas dimensions as label size

### Lines and text positioning
- `lines[]` contains every text element on the plate
- Multiple entries can share the same `line` number (e.g. line 1: ROOM, FANS, SUCTION as separate entries)
- `spacing_top_mm`: distance from top edge of THIS plate to text baseline
- `spacing_left_mm`: distance from left edge of THIS plate to text start
- For layout_guide plates: calculate spacing_left from Horizontal_Segments (e.g. 25, 100, 100, 25)
- For paired sub-labels (DISABLE/ENABLE): position each relative to its column center using 13mm offset if annotated
- `text_size_mm`: estimated font height in mm

### Other fields
- Fields not visible in the analysis → null (not guesses)
- `no_of_holes`: 0 if no holes mentioned
- `label_quantity`: 1 per plate unless stated otherwise
- `label_number`: sequential plate # from Plate Inventory

Return this JSON schema exactly:

{{
  "unit": "mm",
  "total_labels": 0,
  "labels": [
    {{
      "label_number": 1,
      "panel_id": "",
      "section_id": "",
      "grid_type": "plate_divider",
      "label_length_mm": 0,
      "label_height_mm": 0,
      "plate_thickness_mm": null,
      "label_quantity": 1,
      "no_of_holes": 0,
      "hole_size_mm": null,
      "hole_distance_mm": null,
      "lines": [
        {{
          "line": 1,
          "text": "",
          "text_size_mm": 0,
          "spacing_top_mm": 0,
          "spacing_left_mm": 0
        }}
      ]
    }}
  ]
}}

Example for a layout_guide plate (250x20mm) with 2 text lines:
- Line 1: ROOM (centered in zone 1), FANS (centered in zone 2), SUCTION (centered in zone 3)
- Line 2: DISABLE + ENABLE under ROOM, LOW + HIGH under FANS, LOW + HIGH under SUCTION

Example for a plate_divider plate (65x40mm):
- Line 1: ROOM 31 (centered)

Return valid JSON only.
"""


def resolve_input_path(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if path.is_dir():
            path = path / OUTPUT_MD
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path

    results = Path(RESULT_DIR)
    if not results.is_dir():
        print(f"Error: no {RESULT_DIR}/ directory found.", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(results.iterdir(), reverse=True)
    for folder in candidates:
        if folder.is_dir():
            md_path = folder / OUTPUT_MD
            if md_path.exists():
                return md_path

    print(f"Error: no {OUTPUT_MD} found in {RESULT_DIR}/", file=sys.stderr)
    sys.exit(1)


def read_markdown_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    if "---" in content:
        parts = content.split("---", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return content.strip()


def save_specs(output_path: Path, data: dict, source_md: str) -> None:
    output = {
        "source_md": source_md,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def generate_specs(md_path: Path) -> Path:
    markdown = read_markdown_body(md_path)
    prompt = build_specs_prompt(markdown)

    content = call_chat(
        [{"role": "user", "content": prompt}],
        label="specs",
        max_tokens=API_SPECS_MAX_TOKENS,
    )

    parsed = parse_json_response(content)
    output_path = md_path.parent / SPECS_FILENAME

    if parsed.get("error") == "parse_failed":
        save_specs(
            output_path,
            {"error": "parse_failed", "raw_response": parsed.get("raw_response", content)},
            str(md_path),
        )
        print(f"Warning: could not parse JSON. Raw response saved to {output_path}", file=sys.stderr)
        sys.exit(1)

    labels = parsed.get("labels", [])
    if not isinstance(labels, list) or not labels:
        save_specs(
            output_path,
            {"error": "empty_labels", "raw_response": content},
            str(md_path),
        )
        print("Warning: no labels in response.", file=sys.stderr)
        sys.exit(1)

    save_specs(output_path, parsed, str(md_path))
    return output_path


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    md_path = resolve_input_path(arg)

    print(f"Generating specs from {md_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print(
        f"Timeout: read={API_READ_TIMEOUT}s, "
        f"specs_max_tokens={API_SPECS_MAX_TOKENS}, "
        f"thinking={'on' if API_ENABLE_THINKING else 'off'}"
    )

    if not check_api_health():
        sys.exit(1)

    if not warmup_model():
        sys.exit(1)

    try:
        output_path = generate_specs(md_path)
    except Exception as exc:
        print(f"Specs generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = data.get("labels", [])
    total = data.get("total_labels", len(labels))
    print(f"Specs saved to {output_path}")
    print(f"Total labels: {total}")
    for label in labels:
        num = label.get("label_number", "?")
        panel_id = label.get("panel_id", "N/A")
        w = label.get("label_length_mm", "?")
        h = label.get("label_height_mm", "?")
        line_count = len(label.get("lines", []))
        print(f"  - #{num} {panel_id} ({w}x{h}mm): {line_count} text entries")


if __name__ == "__main__":
    main()
