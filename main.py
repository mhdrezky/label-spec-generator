import base64
import os
import sys
from datetime import datetime

from requests.exceptions import ConnectionError, ReadTimeout, RequestException

from api_client import (
    API_MAX_RETRIES,
    API_MAX_TOKENS,
    API_READ_TIMEOUT,
    API_URL,
    MODEL,
    API_CONNECT_TIMEOUT,
    API_ENABLE_THINKING,
    api_post,
    build_api_payload,
    check_api_health,
    describe_empty_response,
    extract_content,
    warmup_model,
)

IMAGE_FILE = "draft.png"
RESULT_DIR = "results"
OUTPUT_FILENAME = "output.md"


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_extraction_prompt() -> str:
    return """
Analyze this label or panel layout image.

Return the result as Markdown only. No JSON.

Describe what you see AND reconstruct a visual preview that mirrors the image layout.
Another system will use this as context to rebuild or edit the layout.

Use millimeters (mm) when dimension annotations are visible.
If no dimension lines exist, write "Dimensions: not annotated" and describe layout proportionally.
If units are unclear, state your assumption explicitly.

Include these sections (use the headings exactly, in this order):

## Layout Overview
- Layout type and purpose (brief)
- Estimated canvas size (width x height of the full drawing sheet)
- Overall structure summary
- Total physical label plates to manufacture (count separately from visual grid rows)

## Grid Specifications
Rigid key-value facts extracted from dimension lines when visible.
Include ONLY fields that apply to this image — omit any field not supported by visible evidence.

Available fields (use as applicable):

- **Unit:** mm (or state actual unit if not mm)
- **Dimensions:** not annotated (use this line instead of numeric fields when no dimension lines exist)
- **Canvas_Width:** (number — full drawing sheet only)
- **Canvas_Height:** (number — full drawing sheet only)
- **Total_Physical_Plates:** (count of separate manufacturable label plates)
- **Horizontal_Segments:** (comma-separated values from dimension lines as drawn — do NOT invent equal splits)
- **Vertical_Segments:** (comma-separated row/section heights if visible)

Rules for this section:
- Read values ONLY from visible dimension annotations — do NOT divide total width by column count
- Report segment values exactly as annotated (e.g. 25, 100, 100, 25) — never replace with equal column widths
- Distinguish canvas (full drawing) from individual plate media sizes (defined per section below)

## Section Definitions
For EACH visually distinct section, define:

- **Section_ID:** (e.g. Section_1, Section_2)
- **Grid_Type:** one of:
  - `plate_divider` — grid lines split the section into MULTIPLE separate physical plates (each cell = one plate)
  - `layout_guide` — grid lines are GUIDES inside ONE plate per row; each horizontal row = one physical plate
- **Plate_Media_Width_mm:** width of ONE physical plate in this section (not the full canvas)
- **Plate_Media_Height_mm:** height of ONE physical plate in this section
- **Plate_Count:** number of physical plates this section produces
- **Rows_In_Section:** visual rows in the drawing (for layout_guide: must equal Plate_Count)
- **Horizontal_Segments:** segment widths for positioning text WITHIN one plate (e.g. 25, 100, 100, 25)
- **Layout_Guides:** describe vertical/horizontal guide lines and what they align (not plate borders)

Critical rules:
- `plate_divider`: one visual row with N columns → N separate plates (e.g. 4 room headers in one row = 4 plates)
- `layout_guide`: N visual rows → N plates; columns inside each row are text zones, NOT separate plates
- Do NOT count dimension annotation lines or extra grid lines as extra plates
- Exclude crossed-out or marked-invalid rows from plate count
- Each section has its OWN plate media dimensions — do not mix section dimensions

## Plate Inventory
List EVERY physical label plate to manufacture, numbered sequentially:

| Plate # | Section | Plate_Width_mm | Plate_Height_mm | Line_Count | Primary_Text |
|---------|---------|----------------|-----------------|------------|--------------|
| 1 | ... | ... | ... | ... | ... |

For each plate, note:
- Which section it belongs to
- Exact plate media width and height (from Section Definitions, NOT canvas size)
- How many text lines on the plate
- Grid type of its section

## Layout Preview
Rebuild the layout visually so it resembles the image when rendered in a Markdown preview.

**Before writing the preview:**
1. Complete Plate Inventory first — preview must match Total_Physical_Plates
2. For `plate_divider` sections: show each plate as a separate cell/box
3. For `layout_guide` sections: show each ROW as one plate boundary; internal columns are guides only

Rules:
- Use SEPARATE preview blocks per section
- For `layout_guide` sections: one HTML table row = one physical plate; do NOT treat columns as separate plates
- Show EVERY text label with correct grouping inside each plate
- Row count in preview must match Plate_Count per section (not inflated by counting guide lines)

**Formatting rules (critical for correct rendering):**

1. **plate_divider section** (multiple plates in one visual row) → Markdown pipe table, each cell = one plate:
   | PLATE 1 text | PLATE 2 text | PLATE 3 text |
   |:------------:|:------------:|:------------:|

2. **layout_guide section, multi-line plate** → one `<tr>` per physical plate:
   <table>
   <tr><td><strong>HEADER_A</strong><br>SUB_A&nbsp;&nbsp;SUB_B</td><td>...</td></tr>
   </table>

3. **layout_guide section, single-line plate** → one `<tr>` per plate with positioned texts:
   <tr><td>TEXT_LEFT</td><td>TEXT_CENTER</td><td>TEXT_RIGHT</td></tr>

4. NEVER put `<br>` inside Markdown pipe table cells
5. Add comment above each block: `<!-- Section_X: grid_type, N plates, WxH mm each -->`

## Grid and Sections
- Reference Section Definitions — repeat grid_type and plate media size per section
- For layout_guide: explain how horizontal segments position text within each plate

## Text Content
- List every visible text label
- For each: text, which plate # it belongs to, line number within that plate, alignment, estimated font size
- For layout_guide plates with multiple texts on one line: list each text with its horizontal zone (left/center/right or mm offset)

## Repeated Patterns
- Pattern name, plate_count, which section, what each plate contains

## Dimensions and Annotations
- All measurement values shown on the drawing, or "none visible"
- Separate canvas dimensions from per-plate media dimensions

Be thorough and factual. Only describe what is visible in the image.
Keep the response concise — avoid repeating the same information across sections.
"""


def call_vision_api(image_b64: str) -> str:
    payload = build_api_payload(
        model=MODEL,
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_extraction_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    timeout = (API_CONNECT_TIMEOUT, API_READ_TIMEOUT)
    last_error: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print(f"Retrying vision request ({attempt}/{API_MAX_RETRIES})...")

            print("Sending vision request (model may take several minutes on first load)...")
            response = api_post(payload, timeout=timeout, label="vision")
            response.raise_for_status()

            data = response.json()
            content = extract_content(data)
            if content is None:
                raise ValueError(describe_empty_response(data))

            print(f"Vision response received ({len(content)} chars).")
            return content
        except ReadTimeout as exc:
            last_error = exc
            print(
                f"Read timeout after {API_READ_TIMEOUT}s "
                f"(attempt {attempt}/{API_MAX_RETRIES})",
                file=sys.stderr,
            )
        except (ConnectionError, ValueError, RequestException) as exc:
            last_error = exc
            print(f"Request failed: {exc} (attempt {attempt}/{API_MAX_RETRIES})", file=sys.stderr)

    raise last_error if last_error else RuntimeError("Vision request failed")


def create_result_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_output_markdown(image_path: str, content: str) -> str:
    extracted_at = datetime.now().isoformat(timespec="seconds")
    header = f"""# Label Layout Analysis

- **Source image:** {image_path}
- **Model:** {MODEL}
- **Extracted at:** {extracted_at}

---

"""
    body = content.strip()
    if not body.startswith("#"):
        return header + body
    return header + body


def save_output(content: str, output_dir: str, image_path: str) -> str:
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(build_output_markdown(image_path, content))
    return output_path


def save_error(output_dir: str, error: str, image_path: str) -> str:
    content = f"""# Label Layout Analysis

- **Source image:** {image_path}
- **Model:** {MODEL}
- **Status:** failed

---

## Error

{error}

- **API URL:** {API_URL}
- **Read timeout:** {API_READ_TIMEOUT}s
- **Max retries:** {API_MAX_RETRIES}
"""
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def extract_metadata(image_path: str) -> str:
    image_b64 = image_to_base64(image_path)
    return call_vision_api(image_b64)


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_FILE

    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting metadata from {image_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print(
        f"Timeout: connect={API_CONNECT_TIMEOUT}s, "
        f"read={API_READ_TIMEOUT}s, retries={API_MAX_RETRIES}, "
        f"thinking={'on' if API_ENABLE_THINKING else 'off'}"
    )

    output_dir = create_result_dir()
    print(f"Output directory: {output_dir}")

    if not check_api_health():
        error_path = save_error(
            output_dir,
            "API tidak merespons. Model LLM mungkin masih loading di server.",
            image_path,
        )
        print(f"Error details saved to {error_path}", file=sys.stderr)
        sys.exit(1)

    if not warmup_model():
        error_path = save_error(
            output_dir,
            "Model warmup gagal. Pastikan model sudah selesai load di server.",
            image_path,
        )
        print(f"Error details saved to {error_path}", file=sys.stderr)
        sys.exit(1)

    try:
        content = extract_metadata(image_path)
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        error_path = save_error(output_dir, str(exc), image_path)
        print(f"Error details saved to {error_path}", file=sys.stderr)
        sys.exit(1)

    if not content:
        print("Warning: empty response from model.", file=sys.stderr)
        sys.exit(1)

    output_path = save_output(content, output_dir, image_path)
    line_count = content.count("\n") + 1

    print(f"Analysis saved to {output_path}")
    print(f"Response length: {len(content)} chars, ~{line_count} lines")


if __name__ == "__main__":
    main()
