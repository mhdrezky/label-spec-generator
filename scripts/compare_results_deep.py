"""Deep compare key runs across models."""

import json
from pathlib import Path


def resolve_run(run: str) -> Path:
    for base in (
        Path("results/current"),
        Path("results/benchmark-models"),
        Path("results/archieve"),
        Path("results"),
    ):
        p = base / run if "/" not in run else base / run.split("/", 1)[1]
        if (p / "specs.json").is_file():
            return p
    raise FileNotFoundError(run)


def load_run(run: str):
    p = resolve_run(run)
    spec = json.loads((p / "specs.json").read_text(encoding="utf-8"))
    content = (
        json.loads((p / "stages/02_content.json").read_text(encoding="utf-8"))
        if (p / "stages/02_content.json").is_file()
        else None
    )
    structure = (
        json.loads((p / "stages/01_structure.json").read_text(encoding="utf-8"))
        if (p / "stages/01_structure.json").is_file()
        else None
    )
    return p, spec, content, structure


def print_drawing_comparison() -> None:
    runs = [
        ("35B archive", "archieve/20260716_112845"),
        ("35B current", "20260716_115809"),
        ("VL-32B", "_Qwen3-VL-32B-Instruct-FP8_20260717_085230"),
        ("9B", "_Qwen3.5-9B-AWQ-BF16-INT8_20260717_092026"),
    ]
    print("=== drawing.png: LLM content vs final ===")
    for name, run in runs:
        p, spec, content, structure = load_run(run)
        print(f"\n{name} ({p.name})")
        if structure:
            print(f"  structure plates: {len(structure.get('plates') or [])}")
        if content:
            for pl in content.get("plates") or []:
                for ln in pl.get("lines") or []:
                    print(
                        f"  content #{pl.get('label_number')}: {ln.get('text')!r} "
                        f"x={ln.get('x_mm')} y={ln.get('y_mm')} size={ln.get('size_mm')}"
                    )
        for lab in spec.get("labels") or []:
            for ln in lab.get("lines") or []:
                print(
                    f"  FINAL   #{lab.get('label_number')}: {ln.get('text')!r} "
                    f"x={ln.get('x_mm')} y={ln.get('y_mm')} size={ln.get('size_mm')}"
                )


def print_image003_comparison() -> None:
    runs = [
        ("35B", "20260716_115827"),
        ("VL-32B", "_Qwen3-VL-32B-Instruct-FP8_20260717_085308"),
        ("9B", "_Qwen3.5-9B-AWQ-BF16-INT8_20260717_092108"),
    ]
    print("\n=== image003: structure gate ===")
    for name, run in runs:
        _, spec, _, structure = load_run(run)
        gate = (structure or {}).get("gate") or {}
        print(
            f"{name}: final_labels={len(spec.get('labels') or [])} "
            f"struct_plates={len((structure or {}).get('plates') or [])} "
            f"cv={gate.get('cv_count')} llm={gate.get('llm_count')} "
            f"trust_cv={gate.get('trust_cv')}"
        )

    print("\n=== image003 plate #5 layout ===")
    for name, run in runs[:2]:
        _, spec, _, _ = load_run(run)
        lab = next(l for l in spec["labels"] if l["label_number"] == 5)
        ys = sorted({round(l["y_mm"], 1) for l in lab["lines"]})
        xs = [round(l["x_mm"], 1) for l in lab["lines"][:6]]
        print(f"{name}: {lab['width_mm']}x{lab['height_mm']} rows y={ys} xs={xs}")


def print_35b_reproducibility() -> None:
    pairs = [
        ("drawing.png", "20260716_115809", "archieve/20260716_112845"),
        ("image003.png", "20260716_115827", "archieve/20260716_112922"),
        ("traffolyte-pdf.png", "20260716_115904", "archieve/20260716_113040"),
    ]
    print("\n=== Qwen3.6-35B reproducibility (same model, different runs) ===")
    for src, run_a, run_b in pairs:
        _, spec_a, content_a, struct_a = load_run(run_a)
        _, spec_b, content_b, struct_b = load_run(run_b)
        same_content = content_a == content_b
        same_struct = struct_a == struct_b
        same_final = spec_a.get("labels") == spec_b.get("labels")
        print(
            f"{src}: structure={same_struct} content={same_content} final={same_final}"
        )


if __name__ == "__main__":
    print_drawing_comparison()
    print_image003_comparison()
    print_35b_reproducibility()
