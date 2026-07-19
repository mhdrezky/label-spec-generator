"""Compare all results/ runs grouped by source image."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

POSITION_FIELDS = ("x_mm", "y_mm", "size_mm")


def model_from_run(name: str) -> str:
    if name.startswith("_") and name.count("_") >= 2:
        ts = name.rsplit("_", 2)
        if len(ts) == 3 and ts[1].isdigit() and ts[2].isdigit():
            return ts[0][1:]
    return "Qwen3.6-35B-A3B-FP8 (assumed)"


def short_model(model: str) -> str:
    return model.rsplit("/", 1)[-1] if "/" in model else model


def label_signature(label: dict) -> tuple:
    lines = label.get("lines") or []
    return (
        label.get("label_number"),
        label.get("width_mm"),
        label.get("height_mm"),
        tuple(
            (
                ln.get("text"),
                ln.get("x_mm"),
                ln.get("y_mm"),
                ln.get("size_mm"),
            )
            for ln in lines
        ),
    )


def load_llm_content(run_dir: Path) -> dict | None:
    content_path = run_dir / "stages" / "02_content.json"
    if not content_path.is_file():
        return None
    return json.loads(content_path.read_text(encoding="utf-8"))


def content_line_summary(content: dict | None) -> list[tuple]:
    if not content:
        return []
    out = []
    for plate in content.get("plates") or []:
        for ln in plate.get("lines") or []:
            out.append(
                (
                    plate.get("label_number"),
                    ln.get("text"),
                    ln.get("x_mm"),
                    ln.get("y_mm"),
                    ln.get("size_mm"),
                )
            )
    return out


def main() -> None:
    root = Path("results")
    rows = []
    for spec_path in sorted(root.rglob("specs.json")):
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        run_dir = spec_path.parent
        rid = run_dir.name
        src = Path(data.get("source_image") or "?").name
        model = short_model(data.get("model") or model_from_run(rid))
        labels = data.get("labels") or []
        warnings = data.get("warnings") or []
        content = load_llm_content(run_dir)
        rows.append(
            {
                "run": rid,
                "model": model,
                "src": src,
                "labels": len(labels),
                "warnings": warnings,
                "signature": tuple(label_signature(l) for l in labels),
                "content_lines": content_line_summary(content),
                "content": content,
            }
        )

    print("=== ALL RUNS ===")
    print(f"{'run':<48} {'model':<30} {'image':<32} lbl warn")
    for r in sorted(rows, key=lambda x: (x["src"], x["run"])):
        print(
            f"{r['run']:<48} {r['model']:<30} {r['src']:<32} "
            f"{r['labels']:>3} {len(r['warnings']):>4}"
        )

    by_src: dict[str, list] = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)

    print("\n=== ANALYSIS BY IMAGE ===")
    for src, items in sorted(by_src.items()):
        print(f"\n--- {src} ({len(items)} runs) ---")
        models = sorted({i["model"] for i in items})
        label_counts = sorted({i["labels"] for i in items})
        signatures = {i["signature"] for i in items}
        content_sets = {tuple(i["content_lines"]) for i in items if i["content_lines"]}

        print(f"  models: {models}")
        print(f"  final label counts: {label_counts}")
        print(f"  distinct final outputs: {len(signatures)}")
        print(f"  distinct LLM content outputs: {len(content_sets) or 'no stages'}")

        if len(label_counts) > 1 or len(signatures) > 1:
            for i in items:
                print(
                    f"    {i['run']} [{i['model']}]: "
                    f"{i['labels']} labels, {len(i['warnings'])} warnings"
                )
                for w in i["warnings"][:3]:
                    print(f"      ! {w}")

        # Compare LLM content vs final for same image across models
        if len(items) >= 2 and content_sets:
            ref = items[0]
            for other in items[1:]:
                if ref["content_lines"] != other["content_lines"]:
                    print(
                        f"  LLM content differs: {ref['run']} vs {other['run']}"
                    )
                if ref["signature"] != other["signature"] and ref["content_lines"] == other["content_lines"]:
                    print(
                        f"  Same LLM content, different final: {ref['run']} vs {other['run']} "
                        "(pipeline/postprocess)"
                    )


if __name__ == "__main__":
    main()
