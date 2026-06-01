# MutaKernel — Supplementary Material

Anonymous supplementary material for the paper
**"MutaKernel: Validating the Validators of LLM-Generated GPU Kernels,"**
released to preserve double-blind review.

## The site

[`index.html`](index.html) is the live page — a single, self-contained page (no build
step) that GitHub Pages serves directly from the repository root. It contains the
abstract, the two workflow figures, and the supplementary material (mutation
operators, EMD Layer-1 static rules, the stress-policy library, and the RQ5 repair
experiment).

`docs/index.md` is the same content in Markdown, kept as an editable source.

## Before publishing

1. Add the two figures to `figures/`:
   - `study-workflow.png` — paper Figure 2 (three-stage study workflow)
   - `mutakernel-workflow.png` — paper Figure 3 (MutaKernel workflow)

   Until they are present, the page shows a labeled placeholder in each figure slot.
2. (Optional) Add raw RQ5 records under `data/`; the field layout is described in
   the RQ5 section.
3. Confirm GitHub Pages is enabled with the source set to the repository root.

## Layout

```
.
├── index.html            # the live page (self-contained: HTML + CSS + JS, web fonts)
├── README.md
├── docs/
│   └── index.md           # Markdown source of the same content
├── figures/
│   ├── study-workflow.png         # add
│   └── mutakernel-workflow.png    # add
└── data/                  # (optional) raw RQ5 records
```
