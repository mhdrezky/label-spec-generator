# Label Spec

- **Source image:** D:\Repositories\label-extractor\sample-data\kiso-handwritten.jpeg
- **Generated at:** 2026-07-21T08:43:41
- **Unit:** mm
- **Total labels:** 2

Value provenance: plain = annotated in the draft, `~` = measured from the drawing via pixel calibration, `*` = computed by the layout resolver — verify `*` values before manufacturing.

## Label 1 — 20 x 216 mm

- **Quantity:** 1
- **Background:** White
- **Text colour:** Black

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| KISO 1.1 | 10 | 36 | 5 | — | — |
| KISO 1.2 | 10 | 72 | 5 | — | — |
| KISO 1.3 | 10 | 108 | 5 | — | — |
| KISO 1.4 | 10 | 144 | 5 | — | — |
| KISO 1.6 | 10 | 180 | 5 | — | — |
| KISO 1.5 | 10 | 216 | 5 | — | — |
| KISO 1.7 | 10 | — | 5 | — | — |

## Label 2 — 20 x 216 mm

- **Quantity:** 1
- **Background:** White
- **Text colour:** Black

| Text | Position from left (mm) | Position from top (mm) | Text size (mm) | Alignment | Bold |
|---|---|---|---|---|---|
| KISO 2.1 | 10 | 36 | 5 | — | — |
| KISO 2.2 | 10 | 72 | 5 | — | — |
| KISO 2.3 | 10 | 108 | 5 | — | — |
| KISO 2.4 | 10 | 144 | 5 | — | — |
| KISO 2.6 | 10 | 180 | 5 | — | — |
| KISO 2.5 | 10 | 216 | 5 | — | — |
| KISO 2.7 | 10 | — | 5 | — | — |

## Warnings

- structure: using LLM plate outlines (CV gate: cv=4, llm=2, trust_cv=False)
- plate #1: width_mm=20 inconsistent with bbox scale (sx=12.80 sy=3.22) — re-inferring from outline
- plate #2: width_mm=20 inconsistent with bbox scale (sx=12.80 sy=3.22) — re-inferring from outline
- label #1 (KISO 1.1): 'KISO 1.7' y_mm=252.0 outside plate (216mm) — dropped
- label #2 (KISO 2.1): 'KISO 2.7' y_mm=252.0 outside plate (216mm) — dropped
- label #1 (KISO 1.1): 'KISO 1.5' (center 216.0mm, size 5.0mm) extends past plate edge (height 216)
- label #2 (KISO 2.1): 'KISO 2.5' (center 216.0mm, size 5.0mm) extends past plate edge (height 216)
