"""Designer-style pipeline orchestrator."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from context import SheetContext
from measure import run_measure
from nodes.decompose import run_decompose
from nodes.dimensions import run_dimensions
from nodes.plate_pipeline import run_all_plates
from nodes.sheet_qc import run_sheet_qc
from nodes.survey import run_survey

STAGE_NAMES = (
    "01_survey",
    "02_dimensions",
    "03_decompose",
    "04_transcribe",
    "05_position",
    "06_size",
    "07_measure",
    "08_qc",
    "09_output",
)

MAX_QC_RETRIES = 1


def _snapshot_stage(stage_dir: str | None, name: str, ctx: SheetContext) -> None:
    if not stage_dir:
        return
    os.makedirs(stage_dir, exist_ok=True)
    payload = copy.deepcopy(ctx.to_spec_dict())
    payload["pipeline_stage"] = name
    payload["plate_regions"] = ctx.plate_regions
    payload["draft_type"] = ctx.draft_type
    payload["material_notes"] = ctx.material_notes
    if ctx.qc_result:
        payload["qc_result"] = ctx.qc_result
    path = os.path.join(stage_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _apply_qc_retry(
    image_path: str | Path,
    ctx: SheetContext,
    retries_used: dict[str, int],
    stage_dir: str | None,
) -> SheetContext:
    qc = ctx.qc_result or {}
    if qc.get("verdict") != "revise":
        return ctx

    fix = qc.get("fix")
    if fix == "decompose" and retries_used.get("decompose", 0) < MAX_QC_RETRIES:
        retries_used["decompose"] = retries_used.get("decompose", 0) + 1
        ctx.warnings.append("QC retry: re-running decompose + per-plate nodes")
        ctx.labels = []
        qc_hint = (ctx.qc_result or {}).get("notes")
        ctx = run_decompose(image_path, ctx, qc_hint=qc_hint)
        _snapshot_stage(stage_dir, "03_decompose", ctx)
        ctx = run_all_plates(image_path, ctx)
        _snapshot_stage(stage_dir, "06_size", ctx)
        spec = ctx.spec_for_measure()
        run_measure(spec, ctx.warnings)
        ctx.labels = spec["labels"]
        _snapshot_stage(stage_dir, "07_measure", ctx)
        return run_sheet_qc(image_path, ctx)

    plate_ids = qc.get("plate_ids") or []
    if fix == "position" and plate_ids and retries_used.get("position", 0) < MAX_QC_RETRIES:
        retries_used["position"] = retries_used.get("position", 0) + 1
        ctx.warnings.append(f"QC retry: re-running position for plates {plate_ids}")
        ctx = run_all_plates(
            image_path, ctx, plate_ids=plate_ids,
            transcribe=False, position=True, size=False,
        )
        spec = ctx.spec_for_measure()
        run_measure(spec, ctx.warnings)
        ctx.labels = spec["labels"]
        _snapshot_stage(stage_dir, "07_measure", ctx)
        return run_sheet_qc(image_path, ctx)

    if fix == "size" and plate_ids and retries_used.get("size", 0) < MAX_QC_RETRIES:
        retries_used["size"] = retries_used.get("size", 0) + 1
        ctx.warnings.append(f"QC retry: re-running size for plates {plate_ids}")
        ctx = run_all_plates(
            image_path, ctx, plate_ids=plate_ids,
            transcribe=False, position=False, size=True,
        )
        spec = ctx.spec_for_measure()
        run_measure(spec, ctx.warnings)
        ctx.labels = spec["labels"]
        _snapshot_stage(stage_dir, "07_measure", ctx)
        return run_sheet_qc(image_path, ctx)

    return ctx


def run_pipeline(
    image_path: str | Path,
    *,
    stage_dir: str | None = None,
) -> SheetContext:
    """Run the full designer pipeline. Returns final SheetContext."""
    image_path = Path(image_path)
    ctx = SheetContext(image_path=str(image_path))

    ctx = run_survey(image_path, ctx)
    _snapshot_stage(stage_dir, "01_survey", ctx)

    ctx = run_dimensions(image_path, ctx)
    _snapshot_stage(stage_dir, "02_dimensions", ctx)

    ctx = run_decompose(image_path, ctx)
    _snapshot_stage(stage_dir, "03_decompose", ctx)

    ctx = run_all_plates(image_path, ctx)
    _snapshot_stage(stage_dir, "06_size", ctx)

    ctx.pre_measure_labels = copy.deepcopy(ctx.labels)
    spec = ctx.spec_for_measure()
    run_measure(spec, ctx.warnings)
    ctx.labels = spec["labels"]
    _snapshot_stage(stage_dir, "07_measure", ctx)

    ctx = run_sheet_qc(image_path, ctx)
    _snapshot_stage(stage_dir, "08_qc", ctx)

    retries_used: dict[str, int] = {}
    for _ in range(MAX_QC_RETRIES):
        if (ctx.qc_result or {}).get("verdict") != "revise":
            break
        ctx = _apply_qc_retry(image_path, ctx, retries_used, stage_dir)
        _snapshot_stage(stage_dir, "08_qc", ctx)

    _snapshot_stage(stage_dir, "09_output", ctx)
    return ctx
