"""Quality-aware benchmark scoring — precision, over-extraction, and dimensions.

Text recall alone is blind to problems you SEE in the editor: duplicated /
extra lines, merged plate dimensions, flattened sizes. This scorer adds them.

    python scripts/eval_quality.py [--root DIR] [--prefix MODEL] [--since TS] [--until TS]

For each image with a ground truth in benchmark/expected/, it finds the latest
matching run under <root>/results/ and reports, per image and in aggregate:

- labels     : predicted label count vs expected (exact-match flag)
- text P/R/F1: MULTISET precision & recall of transcribed lines. Precision drops
               when the model emits extra OR duplicated lines (a number printed
               twice counts once in the truth, twice in the prediction -> P<1).
- lines      : total predicted lines / total expected lines (>1 = over-extraction)
- dims       : % of expected width_mm/height_mm matched within +/-2mm (merged or
               mis-split plates fail this)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DIM_TOL_MM = 2.0


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").upper()).strip()


def label_line_texts(label: dict) -> list[str]:
    return [norm(ln.get("text", "")) for ln in label.get("lines") or [] if norm(ln.get("text", ""))]


def label_text_set(label: dict) -> set[str]:
    return set(label_line_texts(label))


def pair_labels(expected: list[dict], predicted: list[dict]) -> list[tuple[dict, dict]]:
    pairs, remaining = [], list(predicted)
    for exp in expected:
        et = label_text_set(exp)
        best, best_score = None, -1
        for pred in remaining:
            score = len(et & label_text_set(pred))
            if score > best_score:
                best, best_score = pred, score
        if best is not None:
            pairs.append((exp, best))
            remaining.remove(best)
    return pairs


def score_image(expected: dict, pred: dict) -> dict:
    exp_labels = expected.get("labels") or []
    pred_labels = pred.get("labels") or []

    tp = pred_total = exp_total = 0          # multiset text counts
    dim_hit = dim_total = 0
    exp_lines_total = pred_lines_total = 0
    extras: Counter = Counter()

    for exp in exp_labels:
        exp_lines_total += len(label_line_texts(exp))
    for pred_l in pred_labels:
        pred_lines_total += len(label_line_texts(pred_l))

    for exp, pred_l in pair_labels(exp_labels, pred_labels):
        e = Counter(label_line_texts(exp))
        p = Counter(label_line_texts(pred_l))
        inter = sum((e & p).values())
        tp += inter
        pred_total += sum(p.values())
        exp_total += sum(e.values())
        for t, c in (p - e).items():
            extras[t] += c
        for field in ("width_mm", "height_mm"):
            ev = exp.get(field)
            if isinstance(ev, (int, float)) and not isinstance(ev, bool):
                dim_total += 1
                pv = pred_l.get(field)
                if isinstance(pv, (int, float)) and not isinstance(pv, bool) and abs(ev - pv) <= DIM_TOL_MM:
                    dim_hit += 1

    return {
        "exp_labels": len(exp_labels),
        "pred_labels": len(pred_labels),
        "tp": tp, "pred_total": pred_total, "exp_total": exp_total,
        "dim_hit": dim_hit, "dim_total": dim_total,
        "exp_lines": exp_lines_total, "pred_lines": pred_lines_total,
        "extras": extras,
    }


def latest_runs(results_dir: Path, prefix: str, since: int, until: int) -> dict[str, Path]:
    m: dict[str, Path] = {}
    for d in sorted(results_dir.glob(f"{prefix}*")):
        sp = d / "specs.json"
        if not sp.is_file():
            continue
        mt = re.search(r"_(\d{8}_\d{6})$", d.name)
        if mt:
            ts = int(mt.group(1).replace("_", ""))
            if not (since <= ts <= until):
                continue
        try:
            stem = Path(json.loads(sp.read_text(encoding="utf-8"))["source_image"]).stem
        except Exception:
            continue
        m[stem] = d  # sorted ascending -> keep latest
    return m


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="project root containing results/ and benchmark/")
    ap.add_argument("--prefix", default="_Qwen", help="run-dir model prefix filter")
    ap.add_argument("--since", type=int, default=0, help="min run timestamp YYYYMMDDHHMMSS")
    ap.add_argument("--until", type=int, default=99999999999999)
    ap.add_argument("--expected", default=None, help="expected dir (defaults to this repo's benchmark/expected)")
    args = ap.parse_args()

    root = Path(args.root)
    exp_dir = Path(args.expected) if args.expected else DEFAULT_ROOT / "benchmark" / "expected"
    runs = latest_runs(root / "results", args.prefix, args.since, args.until)

    rows = []
    agg = dict(lab_ok=0, lab_n=0, tp=0, pt=0, et=0, dh=0, dt=0, el=0, pl=0)
    all_extras: Counter = Counter()

    for exp_path in sorted(exp_dir.glob("*.json")):
        stem = exp_path.stem
        if stem not in runs:
            continue
        expected = json.loads(exp_path.read_text(encoding="utf-8"))
        pred = json.loads((runs[stem] / "specs.json").read_text(encoding="utf-8"))
        s = score_image(expected, pred)
        rows.append((stem, s))
        agg["lab_ok"] += s["pred_labels"] == s["exp_labels"]
        agg["lab_n"] += 1
        agg["tp"] += s["tp"]; agg["pt"] += s["pred_total"]; agg["et"] += s["exp_total"]
        agg["dh"] += s["dim_hit"]; agg["dt"] += s["dim_total"]
        agg["el"] += s["exp_lines"]; agg["pl"] += s["pred_lines"]
        all_extras.update(s["extras"])

    hdr = f"{'image':<38}{'labels':>9} {'txtP':>6} {'txtR':>6} {'F1':>5} {'lines(p/e)':>11} {'dims':>8}"
    print(hdr)
    print("-" * len(hdr))
    for stem, s in rows:
        p = s["tp"] / s["pred_total"] if s["pred_total"] else 0
        r = s["tp"] / s["exp_total"] if s["exp_total"] else 0
        lab = f"{s['pred_labels']}/{s['exp_labels']}" + ("" if s["pred_labels"] == s["exp_labels"] else "!")
        dims = f"{s['dim_hit']}/{s['dim_total']}" if s["dim_total"] else "-"
        print(f"{stem:<38}{lab:>9} {p*100:5.0f}% {r*100:5.0f}% {f1(p,r)*100:4.0f}% "
              f"{str(s['pred_lines'])+'/'+str(s['exp_lines']):>11} {dims:>8}")

    P = agg["tp"] / agg["pt"] if agg["pt"] else 0
    R = agg["tp"] / agg["et"] if agg["et"] else 0
    print("-" * len(hdr))
    print(f"AGGREGATE ({agg['lab_n']} images):")
    print(f"  label exact:     {agg['lab_ok']}/{agg['lab_n']}")
    print(f"  text precision:  {P*100:.0f}%   recall: {R*100:.0f}%   F1: {f1(P,R)*100:.0f}%")
    print(f"  line ratio p/e:  {agg['pl']}/{agg['el']} = {agg['pl']/agg['el']:.2f}  (>1.0 = over-extraction)")
    if agg["dt"]:
        print(f"  dim +/-2mm:      {agg['dh']}/{agg['dt']} = {100*agg['dh']/agg['dt']:.0f}%")
    top = all_extras.most_common(8)
    if top:
        print(f"  top extra/duplicated lines: {', '.join(f'{t!r}x{c}' for t, c in top)}")


if __name__ == "__main__":
    main()
