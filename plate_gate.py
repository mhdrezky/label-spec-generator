"""Phase 2 — decide whether to trust the OpenCV decompose or fall back to the LLM.

All signals are computable WITHOUT ground truth. The count cross-check uses the
cheap `llm_count` yardstick, which is deliberately noisy (it undercounted a
known-12-plate sheet as 8). So the gate is tuned LOOSE: it only rejects CV on
*gross* disagreement or an obviously empty/sparse detection. Subtler
CV-vs-reality mismatches are left to the QC arbiter (Phase 3), which sees the
image. Erring toward trusting CV here is intentional — a tight tolerance against
a noisy yardstick would reject correct CV output (e.g. image003).
"""

from __future__ import annotations

GATE_MIN_PLATES = 2
# |cv - llm| / llm above this => distrust. Loose on purpose (llm_count is noisy):
# image003 cv=12 vs llm=8 is a 50% gap yet CV is correct, so it must stay under
# the bar; marshall p6 cv=3 vs llm=18 (83%) is a real CV failure and must trip it.
GATE_COUNT_TOL = 0.60
# Backstop against "CV found a few tiny specks": total box area vs sheet area.
GATE_MIN_COVERAGE = 0.05


def _sum_area(boxes) -> float:
    # Sum (not true union); overlaps inflate coverage, which is safe because the
    # value is only used as a *floor* — inflation can only avoid a false reject.
    return float(
        sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in boxes)
    )


def evaluate_gate(plates: list[dict], image_px: dict | None, llm_count) -> dict:
    """Return a decision record: trust_cv plus the signals behind it."""
    w = (image_px or {}).get("width") or 0
    h = (image_px or {}).get("height") or 0
    sheet_area = float(w * h) or 1.0

    cv_count = len(plates)
    boxes = [
        p.get("bbox_px")
        for p in plates
        if isinstance(p.get("bbox_px"), (list, tuple)) and len(p.get("bbox_px")) == 4
    ]
    coverage = _sum_area(boxes) / sheet_area

    reasons: list[str] = []
    trust = True

    if cv_count < GATE_MIN_PLATES:
        trust = False
        reasons.append(f"cv_count {cv_count} < {GATE_MIN_PLATES}")

    gap = None
    if isinstance(llm_count, int) and llm_count > 0:
        gap = abs(cv_count - llm_count) / llm_count
        if gap > GATE_COUNT_TOL:
            trust = False
            reasons.append(
                f"count gap {gap:.0%} (cv={cv_count}, llm={llm_count}) "
                f"> {GATE_COUNT_TOL:.0%}"
            )

    if coverage < GATE_MIN_COVERAGE:
        trust = False
        reasons.append(f"coverage {coverage:.0%} < {GATE_MIN_COVERAGE:.0%}")

    return {
        "trust_cv": trust,
        "cv_count": cv_count,
        "llm_count": llm_count,
        "count_gap": round(gap, 3) if gap is not None else None,
        "coverage": round(coverage, 3),
        "reasons": reasons,
    }
