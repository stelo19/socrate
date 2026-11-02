# Socrate — Replicating and Stress-Testing Socratic AI Tutors

EURECOM semester project, 2026–27 · Stefano Longo, Daniele Priola · Supervisor: Raphaël Troncy

We replicate the **TutorMoments** benchmark (Ai2, working paper *"When Help is Unhelpful: Evaluating AI Tutors for Productive Struggle"*) and then stress-test its evaluation pipeline.

## Upstream sources

| What | Link | Version we use |
|---|---|---|
| Code | https://github.com/allenai/tutormoments | commit `c6e26d7` |
| Dataset | https://huggingface.co/datasets/allenai/tutormoments-preview | revision `3bb0c104d65e29facffca7ba864ef10fa22d0e78` |
| Website | https://tutormoments.allen.ai/ | — |

The upstream repository is **not** copied here. Clone it separately (see `docs/setup_guide.md`).
The dataset is **not** copied here either: it contains de-identified tutoring sessions of minors, and its card asks not to redistribute records. Scripts download it from Hugging Face at the pinned revision.

## Contents

```text
scripts/
  inspect_dataset.py     lists the dataset subsets, row counts and columns at a pinned revision
  recompute_table8.py    recomputes paper Table 8 from the released replays (no API key, no cost)
results/
  dataset_inventory.txt  output of inspect_dataset.py
  table8_recomputed.csv  output of recompute_table8.py (with 95% session-bootstrap CIs)
docs/
  paper_summary.md       full analysis of the paper
  setup_guide.md         step-by-step installation and runs
  reproduction_report.md results and comparison with the paper
```

## Quick start

```bash
# 1. install the upstream benchmark (once), following docs/setup_guide.md
git clone https://github.com/allenai/tutormoments.git
cd tutormoments && git checkout c6e26d7
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
cd ..

# 2. run our scripts with that environment active
python scripts/inspect_dataset.py
python scripts/recompute_table8.py --out results/table8_recomputed.csv
```

## Results so far

- Installation verified: upstream test suite 766 passed, 13 skipped.
- **Table 8 replicated exactly**: all 42 values (7 models × 2 prompts × 3 metrics) match the paper when computed on all 520 moments from the released judgments.
- Session-bootstrap confidence intervals (not in the paper): the prompt effect is robust for every model and metric; several model-to-model orderings claimed in the paper are not separated at 95%.

Details in `docs/reproduction_report.md`.

## Rules for this repo

- **Never commit API keys.** Keys live only in a local `.env`, which is ignored by Git.
- Every result states the upstream commit and dataset revision it was produced with.
