# Label Spec

- **Source image:** D:\Repositories\label-extractor\sample-data\brn94-pdf.png
- **Generated at:** 2026-07-21T07:10:58
- **Unit:** mm
- **Total labels:** 3

Value provenance: plain = annotated in the draft, `~` = measured from the drawing via pixel calibration, `*` = computed by the layout resolver — verify `*` values before manufacturing.

## Label 1 — 70 x 40 mm

- **Quantity:** 1
- **Material:** paper
- **Background:** white
- **Text colour:** black
- **Fixing:** none

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| ROOM 28 | 35 | 20 | 10 | — | — |

## Label 2 — 250 x 20 mm

- **Quantity:** 1
- **Material:** paper
- **Background:** white
- **Text colour:** black
- **Fixing:** none

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| CALLING | 39 | 10 | 5 | — | — |
| FAULT | 125 | 10 | 5 | — | — |
| DEFROST | 211 | 10 | 5 | — | — |

## Label 3 — 250 x 20 mm

- **Quantity:** 1
- **Material:** paper
- **Background:** white
- **Text colour:** black
- **Fixing:** none

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| ROOM | 39 | 10 | 5 | — | — |
| FANS | 125 | 10 | 5 | — | — |
| SUCTION | 211 | 10 | 5 | — | — |
| DISABLE | 23.5 | 15 | 5 | — | — |
| ENABLE | 70 | 15 | 5 | — | — |
| LOW | 101.5 | 15 | 5 | — | — |
| HIGH | 148.5 | 15 | 5 | — | — |
| LOW | 195 | 15 | 5 | — | — |
| HIGH | 234.5 | 15 | 5 | — | — |

## Warnings

- structure: using LLM plate outlines (CV gate: cv=0, llm=3, trust_cv=False)
- plate #2: width_mm=250 inconsistent with bbox scale (sx=3.70 sy=8.20) — re-inferring from outline
- plate #3: width_mm=250 inconsistent with bbox scale (sx=3.70 sy=8.25) — re-inferring from outline
- plate #3: content line_count=6 but got 9 line(s)
