"""Shared pipeline state passed between designer-style nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SheetContext:
    image_path: str
    image_px: dict | None = None
    draft_type: str | None = None
    material_notes: str | None = None
    dimension_annotations: list[dict] = field(default_factory=list)
    plate_regions: list[dict] = field(default_factory=list)
    labels: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    qc_result: dict | None = None
    pre_measure_labels: list[dict] | None = None
    unit: str = "mm"
    # decompose routing (Phase 2): which source produced plate_regions, the cheap
    # LLM plate-count yardstick (computed once, reused by the gate and QC), and
    # the gate's decision record.
    decompose_method: str | None = None
    llm_count: int | None = None
    gate: dict | None = None

    def to_spec_dict(self) -> dict[str, Any]:
        """Output shape compatible with editor and render_md."""
        return {
            "unit": self.unit,
            "image_px": self.image_px,
            "dimension_annotations": self.dimension_annotations,
            "labels": self.labels,
            "warnings": list(self.warnings),
            "total_labels": len(self.labels),
        }

    def spec_for_measure(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "image_px": self.image_px,
            "dimension_annotations": self.dimension_annotations,
            "labels": self.labels,
        }
