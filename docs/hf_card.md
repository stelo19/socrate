---
license: cc-by-4.0
pretty_name: Socrate - TutorMoments with open models
tags:
  - education
  - tutoring
  - llm-evaluation
  - benchmark
language:
  - en
---

# Socrate: TutorMoments with open models

EURECOM semester project (Stefano, Daniele; supervisor Pasquale Lisena).

We replicate and stress-test Ai2's **TutorMoments** benchmark
([code](https://github.com/allenai/tutormoments), [data](https://huggingface.co/datasets/allenai/tutormoments-preview),
[site](https://tutormoments.allen.ai/)) using only open or university-provided models:

| role | model | where it runs |
|---|---|---|
| tutor (compared) | `glm` = glm-5.3-flash | EURECOM LLM gateway |
| tutor (compared) | `qwen3-8b` = Qwen/Qwen3-8B (thinking off) | vLLM or transformers |
| tutor (compared) | `llama-3.1-8b` = meta-llama/Llama-3.1-8B-Instruct | vLLM or transformers |
| simulated student | `glm-student` = glm-5.3-flash with low reasoning, same for every tutor | EURECOM gateway |
| judge + action taxonomy | glm-5.3-flash (judge: default reasoning; taxonomy: low) | EURECOM gateway |

Everything else is Ai2's code, unchanged and pinned (commit `c6e26d7`, dataset revision
`3bb0c104d65e29facffca7ba864ef10fa22d0e78`): tutor and student prompts, 5-turn replay,
3-pass judge, metrics. Our scripts only change **which model answers** (see the top of
`scripts/run_benchmark.py`).

**Status:** pilot done (10 moments, GLM tutor, both prompts: `results/`, `reports/comparison.md`);
full run on 100 moments with the three tutors pending.

Subset: 100 moments (50 scaffolding + 50 rigor), drawn with seed 0 from the 520 released
moments; ids in `subsets/subset_100/subset.json`.

## Contents

```
configs/models.yaml      tutors/students available and how to reach them
configs/socrate_config.yaml  benchmark config: student = glm-student, judge = glm-5.3-flash
scripts/make_subset.py   builds the balanced subset (ids + local copy of the moments)
scripts/run_benchmark.py runs the benchmark with the models of configs/models.yaml
scripts/compare_runs.py  final table: S, R, O per tutor x prompt with 95% CIs
scripts/publish_hf.py    exports code + results (no transcripts) and uploads them here
scripts/test_gateway.py  checks the gateway key and lists the models
results/<run>/           summary.json, config.json, verdicts.jsonl (one line per moment)
reports/                 comparison tables and charts
```

Metrics (same as the paper, `tutormoments.report.aggregate`):
**S** = share of scaffolding moments where the tutor scaffolds and does not over-scaffold;
**R** = same for rigor moments; **O** = 1 - share of moments with over-scaffolding.
CIs are a bootstrap over tutoring sessions.

## How to run

### 1. Install (Python 3.11+)

The full code, with Ai2's benchmark as a pinned submodule, is on GitHub:
https://github.com/stelo19/socrate (the README there explains every step and the repo structure).

```bash
git clone --recursive https://github.com/stelo19/socrate.git && cd socrate
python -m venv .venv && source .venv/bin/activate
pip install -e "./tutormoments[dev,analysis]"   # Ai2's benchmark at commit c6e26d7
pip install -r requirements.txt
pip install vllm          # on the GPU machine, for Qwen and Llama
```

Create `.env` (never commit it):

```
EURECOM_API_KEY=...                               # gateway key (Stefano's access)
EURECOM_BASE_URL=https://llm.tools.eurecom.fr/v1
HF_TOKEN=...                                      # needed for Llama (gated) and for publishing
```

Llama 3.1 is gated: accept the licence on its Hugging Face page with the account of `HF_TOKEN`.
Check the gateway: `python scripts/test_gateway.py glm-5.3-flash`.

### 2. Build the subset

```bash
python scripts/make_subset.py --n 100 --seed 0 --out subsets/subset_100
```

`moments.jsonl` is a local copy of dataset records and is not published; `subset.json` (ids) is.

### 3. Start the open models (one server per model)

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --max-model-len 32768
```

Each 8B model needs about 16 GB of GPU memory for the weights. On a single smaller GPU,
run one server at a time and run that tutor only. Other ports or machines:
set `VLLM_QWEN_URL` / `VLLM_LLAMA_URL` (e.g. `http://gpu-node:8000/v1`).
Without vLLM, the same models run in-process with transformers: use the tutor names
`qwen3-8b-hf` / `llama-3.1-8b-hf` (slower, `--concurrency 1`, needs torch + transformers).

### 4. Pilot (10 moments, check that everything works and measure the tokens)

```bash
python scripts/make_subset.py --n 10 --seed 0 --out subsets/pilot_10
python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
    --tutors glm qwen3-8b llama-3.1-8b --modes plain scaffolding_rigor \
    --data_path subsets/pilot_10 --concurrency 2
```

Check `logs/judge_raw.jsonl` (raw judge answers), that `Action mix` is not 100% neutral, and the
`Total tokens` line of each run.

### 5. Full run

```bash
python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
    --tutors glm qwen3-8b llama-3.1-8b --modes plain scaffolding_rigor \
    --data_path subsets/subset_100 --concurrency 4 --log-file run_100.log
python scripts/compare_runs.py --results results --data_path subsets/subset_100 --out reports/comparison
```

Runs are resumable: if the job stops, run the same command again and finished moments are
skipped. Use `nohup ... &` or `tmux` for long jobs.

### 6. Publish

```bash
python scripts/publish_hf.py --repo <hf-user>/socrate-tutormoments            # builds export/
python scripts/publish_hf.py --repo <hf-user>/socrate-tutormoments --upload   # uploads (private)
```

## Cost and time on the gateway

Measured on the 10-moment pilot (GLM tutor, both prompts), gateway tokens per moment:

| role | tokens per moment |
|---|---|
| tutor (GLM only) | about 30k |
| student (`glm-student`) | about 30k |
| judge | about 8.5k |
| action taxonomy | about 0.5k |

So about 68k per moment with GLM as tutor and about 39k with Qwen or Llama (their tutor
turns run on the GPU). For 100 moments x 2 prompts x 3 tutors: roughly **30M gateway tokens**.
Time: a GLM conversation takes 1-4 minutes (the `scaffolding_rigor` prompt is slower); with
`--concurrency 4` expect about **10 hours** in total. Judge calls run 4 at a time
(`JUDGE_CONCURRENCY`).

## Limitations

- **Self-preference**: GLM is the judge, the student, and one of the tutors. Its score as a
  tutor may be inflated; the Qwen vs Llama comparison is not affected in the same way. A check
  with a different judge on a sample is planned.
- The judge is not the paper's (Claude Opus 4.6), so absolute numbers are not comparable with the
  paper's Table 8; comparisons between tutors under the same judge are.
- 100 moments instead of 520: wider confidence intervals (see `reports/comparison.md`).
- Small adaptations to make open models work, with the method unchanged: `<think>` blocks are
  removed from answers; a markdown fence around the judge's JSON is removed before parsing;
  judge requests are sent as normal requests, 4 at a time (no batch API on the gateway);
  reasoning levels follow the paper's config as closely as the gateway allows (student and
  taxonomy: reasoning_effort low, as the paper disables thinking there; judge: default).

## Credits, licences and citation

This project builds on Ai2's work; nothing of theirs is redistributed here. We publish only per-moment verdicts and aggregate results, no transcripts or dataset records: the dataset comes from tutoring sessions with minors and its card asks not to redistribute some records or attempt re-identification.

- **Benchmark code**: [allenai/tutormoments](https://github.com/allenai/tutormoments), © 2026 The Allen Institute
  for Artificial Intelligence, Apache License 2.0. In the GitHub repository it is a git submodule pinned at commit `c6e26d7`, not a copy. We do not modify it; our changes are applied at runtime by
  `scripts/run_benchmark.py`.
- **Dataset**: [allenai/tutormoments-preview](https://huggingface.co/datasets/allenai/tutormoments-preview),
  CC BY 4.0. Scripts download it at the pinned revision; no records are stored here. Its replays include
  conversations generated with third-party models, subject to those providers' terms.

If you use this work, please cite the original benchmark (BibTeX from the dataset card):

```bibtex
@misc{zhang2026whenhelp,
  author       = {Zhang, Albert and Ross, Alexis and Patel, Kajal and Bernado, Julian and
                  Bowie, Rebecca and Trindade Ribeiro, Ana and Halper, Daniel and
                  Valayaputtur, Haripriya and Andreas, Jacob and Loeb, Susanna and
                  Lucy, Li and Lo, Kyle and Knight, Ryan},
  title        = {When Help is Unhelpful: Evaluating AI Tutors for Productive Struggle},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/allenai/tutormoments-preview}}
}
```
