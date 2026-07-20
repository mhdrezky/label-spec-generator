"""Bootstrap benchmark/expected/*.json and score latest --all run results.

Ground truth is hand-curated from draft images + latest extract output.
Review and refine expected files before treating scores as authoritative.

Usage:
    uv run python scripts/bootstrap_expected.py
    uv run python scripts/bootstrap_expected.py --write-only
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.run_benchmark import score_image  # noqa: E402
from dual_call.postprocess import merge_to_spec  # noqa: E402

EXPECTED_DIR = ROOT / "benchmark" / "expected"
RESULTS_DIR = ROOT / "results"
RUN_PREFIX = "_Qwen3.6-35B-A3B-FP8_20260720_"


def _line(text: str) -> dict:
    return {
        "text": text,
        "x_mm": None,
        "y_mm": None,
        "size_mm": None,
        "alignment": None,
        "bold": None,
        "bbox_px": None,
    }


def _label(
    num: int,
    lines: list[str],
    *,
    width_mm: float | None = None,
    height_mm: float | None = None,
    notes: str | None = None,
) -> dict:
    return {
        "label_number": num,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "quantity": None,
        "material": None,
        "background_color": None,
        "text_color": None,
        "fixing": None,
        "notes": notes,
        "bbox_px": None,
        "holes": [],
        "lines": [_line(t) for t in lines],
    }


def _wrap(labels: list[dict]) -> dict:
    return {
        "unit": "mm",
        "image_px": None,
        "dimension_annotations": [],
        "labels": labels,
    }


def _kiso_strip_lines(col: int) -> list[str]:
    order = [1, 2, 3, 4, 6, 5, 7]
    return [f"KISO {col}.{j}" for j in order]


def _range_lines(start: int, end: int) -> list[str]:
    return [str(n) for n in range(start, end + 1)]


def ground_truth_data() -> dict[str, dict]:
    """Hand-curated expected specs keyed by image stem."""
    brn94 = _wrap([
        _label(1, ["Room 28"], width_mm=70, height_mm=40),
        _label(2, ["Room 29"], width_mm=70, height_mm=40),
        _label(3, ["Room 30"], width_mm=70, height_mm=40),
        _label(4, ["CALLING", "FAULT", "DEFROST"], width_mm=250, height_mm=20),
        _label(5, ["CALLING", "FAULT", "DEFROST"], width_mm=250, height_mm=20),
        _label(6, ["CALLING", "FAULT", "DEFROST"], width_mm=250, height_mm=20),
        _label(
            7,
            [
                "ROOM",
                "FANS",
                "SUCTION",
                "DISABLE",
                "ENABLE",
                "LOW",
                "HIGH",
                "LOW",
                "HIGH",
            ],
            width_mm=250,
            height_mm=20,
        ),
        _label(
            8,
            [
                "ROOM",
                "FANS",
                "SUCTION",
                "DISABLE",
                "ENABLE",
                "LOW",
                "HIGH",
                "LOW",
                "HIGH",
            ],
            width_mm=250,
            height_mm=20,
        ),
        _label(
            9,
            [
                "ROOM",
                "FANS",
                "SUCTION",
                "DISABLE",
                "ENABLE",
                "LOW",
                "HIGH",
                "LOW",
                "HIGH",
            ],
            width_mm=250,
            height_mm=20,
        ),
    ])

    drawing = _wrap([
        _label(1, ["7000-SWC-20001", "Switchboard"], width_mm=280, height_mm=80),
    ])
    drawing2 = _wrap([
        _label(1, ["7000-FEE-20001"], width_mm=37, height_mm=10),
    ])
    drawing3 = _wrap([
        _label(1, ["7000-SEN-20001"], width_mm=70, height_mm=20),
    ])

    handwriting = _wrap([
        _label(1, [], width_mm=160, height_mm=60, notes="blank/orange plate"),
        _label(2, ["DUAL SUPPLY", "FED FROM MSSB-S/SB"], width_mm=120, height_mm=60),
        _label(3, ["DUAL SUPPLY", "FED FROM MSSB-S/SB"], width_mm=120, height_mm=60),
        _label(4, ["DUAL SUPPLY", "FED FROM MSSB-S/SB"], width_mm=120, height_mm=60),
        _label(5, ["MSSB-S/B CONTROL", "CB"], width_mm=120, height_mm=None),
    ])

    kiso2 = _wrap([
        _label(
            1,
            ["FED FROM MCC1 TIER 1", "STP008-2000-SLV-003 XA.02"],
            width_mm=200,
            height_mm=35,
        ),
    ])

    kiso3 = _wrap([
        _label(
            1,
            ["FED FROM MCC1 TIER 1", "STP008-2000-SLV-003 XA.02"],
            width_mm=200,
            height_mm=35,
        ),
        _label(2, ["POWER MONITOR"], width_mm=100, height_mm=17),
    ])

    kiso = _wrap([
        _label(1, _kiso_strip_lines(1), width_mm=20, height_mm=216),
        _label(2, _kiso_strip_lines(2), width_mm=20, height_mm=216),
    ])

    marshall_p1 = _wrap([
        _label(
            1,
            [
                "M P E",
                "custom switchboards",
                "SMITHFIELD NSW 2164 (02) 9721 1698",
                "SWITCHBOARD BUILT TO AS/NZS 61439.1/2 : 2016",
                "AS/NZS 3000 : 2018",
                "SEGREGATION : FORM 3b/3bih",
                "DEGREE OF PROTECTION : IP42",
                "RATED : 415v 50Hz",
                "FAULT LEVEL : 40kA / 1 sec",
                "DATE OF MANUFACTURE : JULY 2026",
                "REFERENCE : M6339",
            ],
            width_mm=150,
            height_mm=70,
        ),
        _label(2, ["MAIN SWITCHBOARD 1"], width_mm=130, height_mm=30),
        _label(3, ["SERVICE PROTECTION", "DEVICE"], width_mm=80, height_mm=20),
        _label(4, ["PRIVATE METERING", "TOWER 1"], width_mm=80, height_mm=20),
        _label(5, ["HOUSE SERVICES", "SECTION"], width_mm=80, height_mm=20),
        _label(6, ["HOUSE SERVICES", "TOWER 1 SECTION"], width_mm=80, height_mm=20),
        _label(7, ["HOUSE SERVICES", "TOWER 2 SECTION"], width_mm=80, height_mm=20),
        _label(8, ["SAFETY SERVICES", "TOWER 1 SECTION"], width_mm=80, height_mm=20),
        _label(9, ["EV SECTION", "TOWER 2"], width_mm=80, height_mm=20),
        _label(10, ["SOLAR", "SECTION"], width_mm=80, height_mm=20),
        _label(11, ["RETAIL METERING", "SECTION"], width_mm=80, height_mm=20),
        _label(
            12,
            ["ACCREDITED METERING", "PROVIDER METERING", "CT'S MOUNTED BEHIND"],
            width_mm=80,
            height_mm=30,
        ),
        _label(
            13,
            ["WARNING", "TO BE ACCESSED BY", "AUTHORISED PERSONNEL", "ONLY"],
            width_mm=80,
            height_mm=30,
        ),
        _label(14, ["SURGE DIVERTER"], width_mm=70, height_mm=15),
        _label(15, ["SURGE DIVERTER", "MOUNTED BEHIND"], width_mm=80, height_mm=20),
        _label(
            16,
            ["ACCREDITED METERING", "PROVIDER EQUIPMENT", "MOUNTED BEHIND"],
            width_mm=80,
            height_mm=30,
        ),
    ])

    amp_ratings = [
        "MAX/250A",
        "225/250A",
        "175/250A",
        "200/250A",
        "250/250A",
        "48/80A",
        "MAX/400A",
        "800/800A",
        "280/400A",
        "1250/1250A",
        "320/400A",
        "80/80A",
        "128/160A",
        "162/250A",
        "40/80A",
        "200/400A",
        "80/160A",
        "1000/1000A",
        "144/160A",
        "188/250A",
        "96/160A",
        "68/80A",
    ]
    marshall_p7_labels = [
        _label(
            1,
            [
                "DIGITAL POWER METER 20",
                "MONITORS LIFT #T2-2",
                "CIRCUITS 61 TO 63",
            ],
            width_mm=70,
            height_mm=20,
        ),
        _label(
            2,
            [
                "DIGITAL POWER METER 21",
                "MONITORS LIFT #T2-3",
                "CIRCUITS 64 TO 66",
            ],
            width_mm=70,
            height_mm=20,
        ),
        _label(
            3,
            [
                "DIGITAL POWER METER 22",
                "MONITORS LIFT #T1-2",
                "CIRCUITS 67 TO 69",
            ],
            width_mm=70,
            height_mm=20,
        ),
        _label(
            4,
            ["DIGITAL POWER METER 23", "MONITORS DB-FC", "CIRCUITS 70 TO 72"],
            width_mm=70,
            height_mm=20,
        ),
        _label(
            5,
            [
                "DIGITAL POWER METER 1 EV LOAD",
                "MONITORS MAIN SWITCHBOARD",
                "CIRCUITS 1 TO 72",
            ],
            width_mm=90,
            height_mm=20,
        ),
        _label(
            6,
            [
                "DIGITAL POWER METER 2 EV SOLAR",
                "MONITORS SOLAR SYSTEM",
                "CIRCUIT 6",
            ],
            width_mm=90,
            height_mm=20,
        ),
    ]
    for i, rating in enumerate(amp_ratings, start=7):
        marshall_p7_labels.append(_label(i, [rating], width_mm=30, height_mm=15))
    marshall_p7_labels.extend(
        [
            _label(29, _range_lines(12, 20), width_mm=162, height_mm=15),
            _label(30, _range_lines(21, 29), width_mm=162, height_mm=15),
            _label(31, _range_lines(43, 57), width_mm=270, height_mm=15),
            _label(32, _range_lines(58, 72), width_mm=270, height_mm=15),
        ]
    )
    marshall_p7 = _wrap(marshall_p7_labels)

    mla = _wrap([
        _label(1, ["BELMONT", "PINHOLE", "REJECTS"], width_mm=160, height_mm=80),
        _label(2, ["GEOMETRY", "REJECTS"], width_mm=160, height_mm=34),
        _label(3, ["GEOMETRY", "REJECTS"], width_mm=160, height_mm=34),
        _label(4, ["GEOMETRY", "REJECTS"], width_mm=160, height_mm=34),
        _label(5, ["PINHOLE", "REJECTS"], width_mm=160, height_mm=34),
        _label(6, ["PINHOLE", "REJECTS"], width_mm=160, height_mm=34),
        _label(7, ["REJECTS"], width_mm=60, height_mm=8),
        _label(8, ["REJECTS"], height_mm=8),
        _label(9, ["REJECTS"], height_mm=8),
    ])

    traffolyte = _wrap([
        _label(1, ["GENERATOR SUPPLY", "ISOLATOR", "(100/250A)"], width_mm=140, height_mm=40),
        _label(
            2,
            ["MAIN ISOLATOR", "NON ESSENTIAL POWER", "(150/160A)"],
            width_mm=100,
            height_mm=40,
        ),
        _label(
            3,
            ["MAIN ISOLATOR", "ESSENTIAL LIGHTING", "(80/100A)"],
            width_mm=100,
            height_mm=40,
        ),
        _label(
            4,
            ["MAIN ISOLATOR", "ESSENTIAL POWER", "(80/100A)"],
            width_mm=100,
            height_mm=40,
        ),
        _label(5, ["NON ESSENTIAL", "POWER CHASSIS"], width_mm=100, height_mm=20),
        _label(6, ["ESSENTIAL", "POWER CHASSIS"], width_mm=100, height_mm=20),
        _label(7, ["ESSENTIAL", "LIGHTING CHASSIS"], width_mm=100, height_mm=20),
        _label(8, ["NON ESSENTIAL", "POWER METER"], width_mm=100, height_mm=20),
        _label(9, ["ESSENTIAL POWER", "METER"], width_mm=100, height_mm=20),
        _label(10, ["ESSENTIAL LIGHTING", "METER"], width_mm=100, height_mm=20),
        _label(11, _range_lines(73, 84), width_mm=20, height_mm=215),
        _label(12, _range_lines(85, 96), width_mm=20, height_mm=215),
        _label(13, _range_lines(97, 108), width_mm=20, height_mm=215),
        _label(14, _range_lines(109, 120), width_mm=20, height_mm=215),
        _label(15, ["SPARE"], width_mm=100, height_mm=20),
    ])

    return {
        "brn94-pdf": brn94,
        "drawing": drawing,
        "drawing2": drawing2,
        "drawing3": drawing3,
        "handwriting": handwriting,
        "kiso-handwritten-2": kiso2,
        "kiso-handwritten-3": kiso3,
        "kiso-handwritten": kiso,
        "marshall-ave-st-leonards-msb-1-pdf-p1": marshall_p1,
        "marshall-ave-st-leonards-msb-1-pdf-p7": marshall_p7,
        "mla-white-red-pdf": mla,
        "traffolyte-pdf": traffolyte,
    }


def write_expected_files() -> list[Path]:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, payload in ground_truth_data().items():
        path = EXPECTED_DIR / f"{stem}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)
    return written


def latest_run_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for run in sorted(RESULTS_DIR.glob(f"{RUN_PREFIX}*")):
        spec_path = run / "specs.json"
        if not spec_path.is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        stem = Path(spec["source_image"]).stem
        mapping[stem] = run
    return mapping


def load_run_stages(run_dir: Path) -> tuple[dict, dict]:
    """Re-merge cached stages with current post-process (no new LLM calls)."""
    structure = json.loads(
        (run_dir / "stages" / "01_structure.json").read_text(encoding="utf-8")
    )
    content = json.loads(
        (run_dir / "stages" / "02_content.json").read_text(encoding="utf-8")
    )
    image_px = structure.get("image_px") or content.get("image_px")
    image_path: str | None = None
    if not image_px or (run_dir / "specs.json").is_file():
        specs = json.loads((run_dir / "specs.json").read_text(encoding="utf-8"))
        image_px = image_px or specs.get("image_px") or {}
        image_path = specs.get("source_image")
    warnings: list[str] = []
    measured = merge_to_spec(
        structure, content, image_px, warnings, image_path=image_path
    )
    raw_labels = []
    for lab in measured.get("labels") or []:
        raw_lab = copy.deepcopy(lab)
        for ln in raw_lab.get("lines") or []:
            ln.pop("measured_fields", None)
        raw_labels.append(raw_lab)
    raw = {
        "unit": measured.get("unit"),
        "image_px": measured.get("image_px"),
        "dimension_annotations": measured.get("dimension_annotations"),
        "labels": raw_labels,
    }
    return raw, measured


def parse_pct(score_str: str) -> float | None:
    m = re.match(r"(\d+)%", score_str)
    return int(m.group(1)) if m else None


def parse_label_ratio(score_str: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)/(\d+)", score_str)
    return (int(m.group(1)), int(m.group(2))) if m else None


def score_latest_runs() -> None:
    runs = latest_run_map()
    rows: list[tuple[str, dict]] = []

    label_hit = label_total = 0
    text_hit = text_total = 0
    pos_hit = pos_total = 0
    dim_hit = dim_total = 0

    stems = sorted({*ground_truth_data(), "image003"})
    for stem in stems:
        expected_path = EXPECTED_DIR / f"{stem}.json"
        if not expected_path.is_file():
            continue
        if stem not in runs:
            print(f"  skip {stem}: no matching run in {RUN_PREFIX}*", file=sys.stderr)
            continue
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        raw, measured = load_run_stages(runs[stem])
        score = score_image(expected, raw, measured)
        rows.append((f"{stem}.png", score))

        lr = parse_label_ratio(score["labels"])
        if lr:
            label_hit += 1 if lr[0] == lr[1] else 0
            label_total += 1
        tm = re.match(r"(\d+)% \((\d+)/(\d+)\)", score["text"])
        if tm:
            text_hit += int(tm.group(2))
            text_total += int(tm.group(3))
        pm = re.match(r"(\d+)% \((\d+)/(\d+)\)", score["position"])
        if pm:
            pos_hit += int(pm.group(2))
            pos_total += int(pm.group(3))
        dm = re.match(r"(\d+)% \((\d+)/(\d+)\)", score.get("dimensions", ""))
        if dm:
            dim_hit += int(dm.group(2))
            dim_total += int(dm.group(3))

    if not rows:
        print("No scores computed.", file=sys.stderr)
        sys.exit(1)

    name_width = max(len(name) for name, _ in rows)
    header = (
        f"{'image'.ljust(name_width)}  {'labels':<14} {'text':<14} "
        f"{'position':<14} {'dimensions':<14} {'null':<14}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, score in rows:
        print(
            f"{name.ljust(name_width)}  {score['labels']:<14} {score['text']:<14} "
            f"{score['position']:<14} {score.get('dimensions', 'n/a'):<14} "
            f"{score['null']:<14}"
        )

    print()
    print(f"=== Aggregate ({label_total} samples with ground truth) ===")
    if label_total:
        print(f"Label count match:  {100 * label_hit / label_total:.0f}% ({label_hit}/{label_total} images)")
    if text_total:
        print(f"Text match:         {100 * text_hit / text_total:.0f}% ({text_hit}/{text_total} lines)")
    if pos_total:
        print(f"Position ±2mm:      {100 * pos_hit / pos_total:.0f}% ({pos_hit}/{pos_total} fields)")
    if dim_total:
        print(f"Stated dimensions: {100 * dim_hit / dim_total:.0f}% ({dim_hit}/{dim_total} fields)")
    print()
    print("Note: ground truth is draft-curated; review benchmark/expected/*.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()

    written = write_expected_files()
    print(f"Wrote {len(written)} expected files to {EXPECTED_DIR.relative_to(ROOT)}/")
    if not args.write_only:
        score_latest_runs()


if __name__ == "__main__":
    main()
