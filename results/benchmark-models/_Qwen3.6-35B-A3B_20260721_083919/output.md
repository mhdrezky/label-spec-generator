# Label Spec

- **Source image:** D:\Repositories\label-extractor\sample-data\drawing2.png
- **Generated at:** 2026-07-21T08:39:24
- **Unit:** mm
- **Total labels:** 1

Value provenance: plain = annotated in the draft, `~` = measured from the drawing via pixel calibration, `*` = computed by the layout resolver — verify `*` values before manufacturing.

## Label 1 — 37 x 10 mm

- **Quantity:** 1
- **Material:** Traffolyte
- **Background:** Yellow
- **Text colour:** Black
- **Fixing:** Self-adhesive

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| 7000-FEE-20001 | 18.5 | 5 | 3 | — | — |

## Warnings

- structure: using LLM plate outlines (CV gate: cv=2, llm=1, trust_cv=False)
- plate #1: width_mm=37 inconsistent with bbox scale (sx=6.84 sy=14.50) — re-inferring from outline
- plate #1: text '7000-FEE-20001' too wide at 4mm — reduced to 3mm
