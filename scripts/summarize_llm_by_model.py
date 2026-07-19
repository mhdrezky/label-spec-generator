"""Summarize LLM stage output per model across all runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def model_from_run(name: str, spec: dict) -> str:
    m = spec.get("model")
    if m:
        return m.rsplit("/", 1)[-1]
    if name.startswith("_") and name.count("_") >= 2:
        ts = name.rsplit("_", 2)
        if len(ts) == 3 and ts[1].isdigit() and ts[2].isdigit():
            return ts[0][1:]
    return "Qwen3.6-35B-A3B-FP8"


def load_stages(run_dir: Path) -> tuple[dict | None, dict | None]:
    structure = content = None
    sp = run_dir / "stages/01_structure.json"
    cp = run_dir / "stages/02_content.json"
    if sp.is_file():
        structure = json.loads(sp.read_text(encoding="utf-8"))
    if cp.is_file():
        content = json.loads(cp.read_text(encoding="utf-8"))
    return structure, content


def structure_summary(structure: dict | None) -> str:
    if not structure:
        return "no structure"
    if structure.get("error") == "parse_failed" or (
        not structure.get("plates") and structure.get("draft_type") is None
    ):
        return "INVALID JSON / empty fallback"
    plates = structure.get("plates") or []
    gate = structure.get("gate") or {}
    cv = gate.get("cv_count")
    llm = gate.get("llm_count")
    trust = gate.get("trust_cv")
    src = "CV" if trust else "LLM"
    return f"{len(plates)} plates ({src}; cv={cv}, llm={llm})"


def content_summary(content: dict | None) -> str:
    if not content:
        return "no content"
    if content.get("error") == "parse_failed":
        return "INVALID JSON"
    lines = 0
    plates_with_lines = 0
    for pl in content.get("plates") or []:
        pl_lines = pl.get("lines") or []
        if pl_lines:
            plates_with_lines += 1
        lines += len(pl_lines)
    return f"{lines} lines on {plates_with_lines} plates"


def main() -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)

    for spec_path in sorted(Path("results").rglob("specs.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        run_dir = spec_path.parent
        rid = run_dir.name
        src = Path(spec.get("source_image") or "?").name
        model = model_from_run(rid, spec)
        structure, content = load_stages(run_dir)
        by_model[model].append(
            {
                "image": src,
                "run": rid,
                "structure": structure_summary(structure),
                "content": content_summary(content),
                "final_labels": len(spec.get("labels") or []),
                "warnings": len(spec.get("warnings") or []),
            }
        )

    # One row per image per model (latest run if duplicates)
    for model in sorted(by_model.keys()):
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model}")
        print("=" * 60)
        seen: set[str] = set()
        for row in sorted(by_model[model], key=lambda r: r["image"]):
            if row["image"] in seen:
                continue
            seen.add(row["image"])
            print(f"\n  {row['image']}")
            print(f"    structure LLM: {row['structure']}")
            print(f"    content LLM:   {row['content']}")
            print(f"    final labels:  {row['final_labels']}  (warnings: {row['warnings']})")


if __name__ == "__main__":
    main()
