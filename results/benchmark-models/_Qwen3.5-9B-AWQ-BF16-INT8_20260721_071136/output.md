# Label Spec

- **Source image:** D:\Repositories\label-extractor\sample-data\handwriting.jpg
- **Generated at:** 2026-07-21T07:11:59
- **Unit:** mm
- **Total labels:** 3

Value provenance: plain = annotated in the draft, `~` = measured from the drawing via pixel calibration, `*` = computed by the layout resolver — verify `*` values before manufacturing.

## Label 1 — 110 x 60 mm

- **Quantity:** 1
- **Material:** white blank material
- **Background:** white
- **Text colour:** black
- **Fixing:** glue

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| IN ORANGE BOLD | 55 | 30 | 9 | — | — |

## Label 2 — 110 x 60 mm

- **Quantity:** 1
- **Material:** white blank material
- **Background:** white
- **Text colour:** black
- **Fixing:** glue

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| DRAIN WORK | 55 | 30 | 10 | — | — |
| FAN FILM WEB 4/50 | 55 | 45 | 8 | — | — |

## Label 3 — 120 x 90 mm

- **Quantity:** 1
- **Material:** white blank material
- **Background:** white
- **Text colour:** black
- **Fixing:** glue

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| ADDED 5/25 COMB | 60 | 45 | 10 | — | — |
| CR | 60 | 60 | 8 | — | — |

## Warnings

- structure: using LLM plate outlines (CV gate: cv=6, llm=3, trust_cv=False)
- plate #1: width_mm=160 inconsistent with dimension span sum 110mm — using tiled sum
- plate #1: width_mm=110.0 inconsistent with bbox scale (sx=0.53 sy=1.28) — re-inferring from outline
- plate #1: text 'IN ORANGE BOLD' too wide at 160mm — reduced to 9mm
- plate #2: width_mm=110 inconsistent with bbox scale (sx=0.53 sy=1.28) — re-inferring from outline
