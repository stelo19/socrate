# Socrate — Replicating and Stress-Testing Socratic AI Tutors

EURECOM semester project, 2026–27 · Stefano Longo, Daniele Priola · Supervisor: Pasquale Lisena

We replicate Ai2's **TutorMoments** benchmark (working paper *"When Help is Unhelpful: Evaluating AI Tutors
for Productive Struggle"*) and run it with open and university-provided models instead of commercial APIs.

| What | Link | Version we use |
|---|---|---|
| Benchmark code (Ai2) | https://github.com/allenai/tutormoments — included here as the submodule `tutormoments/` | commit `c6e26d7` |
| Dataset | https://huggingface.co/datasets/allenai/tutormoments-preview | revision `3bb0c104d65e29facffca7ba864ef10fa22d0e78` |
| Our results on Hugging Face | https://huggingface.co/datasets/priolseurec/socrate-tutormoments | — |
| Website | https://tutormoments.allen.ai/ | — |

**What the benchmark does, in one paragraph.** Each of the 520 *moments* is a real K-12 math tutoring
conversation cut at a point where expert teachers said the tutor should either **scaffold** (support a stuck
student) or push for **rigor** (make a succeeding student justify or go further). The tutor model under test
continues the conversation for 5 turns with a simulated student; an LLM judge then decides what the tutor did.
Three metrics, all *higher is better*: **S** (appropriate scaffolding), **R** (appropriate rigor),
**O** (avoids over-scaffolding, i.e. not doing the student's thinking). Details: [`docs/how_it_works.md`](docs/how_it_works.md).

---

## Quick start

Tested on Ubuntu (WSL2 on Windows) with Python 3.12–3.13.

### 1. Clone with the Ai2 submodule and install

```bash
git clone --recursive https://github.com/stelo19/socrate.git
cd socrate
# already cloned without --recursive?  ->  git submodule update --init

python3 -m venv .venv
source .venv/bin/activate
pip install -e "./tutormoments[dev,analysis]"   # the benchmark, at the pinned commit
pip install -r requirements.txt                 # our scripts' dependencies
```

### 2. Keys

Create `.env` in the repo root (it is git-ignored, never commit it):

```
EURECOM_API_KEY=sk-...                            # EURECOM LLM gateway (GLM)
EURECOM_BASE_URL=https://llm.tools.eurecom.fr/v1
HF_TOKEN=hf_...                                   # only for gated models (Llama) and for publishing
```

Check the gateway: `python scripts/test_gateway.py glm-5.3-flash`

### 3. Pick the moments

```bash
python scripts/make_subset.py --n 10  --seed 0 --out subsets/pilot_10     # quick pilot
python scripts/make_subset.py --n 100 --seed 0 --out subsets/subset_100   # main run
```

Balanced (half scaffolding, half rigor), reproducible from the seed. `moments.jsonl` is a local copy of
dataset records (git-ignored); `subset.json` lists the ids and is committed.

### 4. Run tutors

```bash
# GLM as tutor (gateway only, works on a laptop)
python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
    --tutors glm --modes plain scaffolding_rigor --data_path subsets/pilot_10 --concurrency 2

# small open models on a laptop, through Ollama (CPU is fine)
python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
    --tutors llama3.2-3b-ollama qwen2.5-3b-ollama --modes plain --data_path subsets/pilot_10 --concurrency 3

# 8B open models on a GPU, through vLLM (start the servers first, see below)
python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
    --tutors glm qwen3-8b llama-3.1-8b --modes plain scaffolding_rigor \
    --data_path subsets/subset_100 --concurrency 4 --log-file run_100.log
```

Runs are resumable: re-running the same command on the same day skips finished moments.
Each run writes `results/<tutor>_<prompt>_<subset>_<date>/`.

GPU servers (one per model, ~16 GB of GPU memory each for 8B; on a small GPU run one at a time):

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --max-model-len 32768   # accept the licence on HF first
```

Ollama (laptop): install it, then `ollama pull llama3.2:3b` and `ollama pull qwen2.5:3b`.

### 5. Compare

```bash
python scripts/compare_runs.py --data_path subsets/pilot_10 --out reports/comparison_10
```

Writes `reports/comparison_10.md`, `.csv` and `.png`: S, R, O per tutor × prompt with 95% session-bootstrap intervals,
computed only on the moments of the given subset.

### 6. Publish verdicts and summaries to Hugging Face (optional)

```bash
python scripts/publish_hf.py --repo <hf-user>/socrate-tutormoments --runs "*pilot_10*"           # builds export/
python scripts/publish_hf.py --repo <hf-user>/socrate-tutormoments --runs "*pilot_10*" --upload  # uploads
```

---

## Repository structure

```text
socrate/
├── README.md                 this file
├── requirements.txt          our dependencies (the benchmark itself comes from the submodule)
├── configs/
│   ├── models.yaml           which models can play tutor/student and where they run; judge request options
│   ├── socrate_config.yaml   benchmark config: student = glm-student, judge + taxonomy = glm-5.3-flash
│   └── ollama_config.yaml    legacy config for scripts/legacy/run_ollama.py
├── scripts/                  our code (table below); scripts/legacy/ = superseded versions
├── tutormoments/             Ai2's benchmark, git submodule pinned at c6e26d7 (unmodified)
│   └── src/tutormoments/
│       ├── prompts/          ALL the prompts: tutor/, student/, scorer/, taxonomy/
│       ├── conversation.py   5-turn replay loop (tutor <-> simulated student)
│       ├── scoring.py        the 3-pass LLM judge
│       ├── report.py         S, R, O and KL
│       └── default_config.yaml   the paper's configuration
├── subsets/<name>/subset.json    ids of the moments of each subset (+ local moments.jsonl, not committed)
├── results/
│   ├── <run>/summary.json    metrics, action mix, tokens per role (committed)
│   ├── <run>/config.json     settings of the run (committed)
│   ├── <run>/transcripts/, scores/, taxonomy/   per-moment files (local only: sessions with minors)
│   ├── table8_recomputed.csv paper Table 8 recomputed, with confidence intervals
│   └── dataset_inventory.txt
├── reports/                  comparison tables and charts (comparison_10.md / .csv / .png, ...)
├── docs/
│   ├── how_it_works.md       pipeline, judge passes, metrics, our changes, how to change models/prompts
│   ├── paper_summary.md      analysis of the paper
│   ├── reproduction_report.md    Table 8 replication
│   ├── setup_guide.md        first installation guide (Oct 2026)
│   └── hf_card.md            the page of our Hugging Face repo
├── logs/                     raw judge answers (judge_raw.jsonl), local only
└── .env                      your keys (git-ignored)
```

### Scripts

| Script | What it does |
|---|---|
| `run_benchmark.py` | Runs the benchmark with the models of `configs/models.yaml` (gateway, vLLM, Ollama or transformers). Ai2's loop, prompts, judge and metrics are used unchanged. |
| `make_subset.py` | Builds a balanced, seeded subset of moments. |
| `compare_runs.py` | Final table S/R/O per tutor × prompt with session-bootstrap CIs, plus a chart. |
| `publish_hf.py` | Exports code, summaries and per-moment verdicts (no conversations) and uploads them to Hugging Face. |
| `rejudge_benchmark.py` | Re-judges the paper's published conversations with another judge (e.g. GLM) and measures agreement with the paper's judge. |
| `recompute_table8.py` | Recomputes the paper's Table 8 from the released judgments (no API calls). |
| `inspect_dataset.py` | Lists the dataset's subsets, row counts and columns. |
| `show_replay.py` | Prints one conversation: real context, cut point, generated turns. |
| `test_gateway.py` | Checks the gateway key and lists the available models. |
| `test_thinking.py` | Tests which request option turns GLM's hidden reasoning down on the gateway. |
| `legacy/run_ollama.py` | Superseded by `run_benchmark.py` (kept for reference). |

## Where the prompts are

All prompts are Ai2's files, in the submodule (links at the pinned commit):

| Role | File |
|---|---|
| Tutor, `plain` prompt | [`prompts/tutor/plain.txt`](https://github.com/allenai/tutormoments/blob/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/tutor/plain.txt) |
| Tutor, `scaffolding_rigor` prompt | [`prompts/tutor/scaffolding_rigor.txt`](https://github.com/allenai/tutormoments/blob/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/tutor/scaffolding_rigor.txt) |
| Simulated student | [`prompts/student/`](https://github.com/allenai/tutormoments/tree/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/student) |
| Judge, pass 1 (annotate) | [`prompts/scorer/annotate/scaffolding.md`](https://github.com/allenai/tutormoments/blob/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/scorer/annotate/scaffolding.md) |
| Judge, pass 2 (decompose) | [`prompts/scorer/decompose/`](https://github.com/allenai/tutormoments/tree/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/scorer/decompose) |
| Judge, pass 3 (structure) | [`prompts/scorer/structure/`](https://github.com/allenai/tutormoments/tree/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/scorer/structure) |
| Action taxonomy | [`prompts/taxonomy/classify_actions.md`](https://github.com/allenai/tutormoments/blob/c6e26d7d68fd89768bbeea0a600d3bf200d6cabb/src/tutormoments/prompts/taxonomy/classify_actions.md) |

Locally they are under `tutormoments/src/tutormoments/prompts/`. They are loaded by the installed package, so
our scripts never copy or edit them.

## Results so far

- **Table 8 of the paper replicated exactly** from the released judgments (all 42 values), with session-bootstrap
  confidence intervals: [`docs/reproduction_report.md`](docs/reproduction_report.md).
- **Pilot, 10 moments, student and judge GLM** (`reports/comparison_10.md`):

| Tutor | Prompt | S | R | O |
|---|---|---|---|---|
| GLM (glm-5.3-flash) | plain | 80% | 60% | 70% |
| GLM (glm-5.3-flash) | scaffolding_rigor | 20% | 60% | 50% |
| Qwen2.5 3B Instruct (Ollama) | plain | 40% | 40% | 50% |
| Llama 3.2 3B Instruct (Ollama) | plain | 40% | 40% | 40% |

  10 moments give very wide intervals (see the file): these are directions to check on 100 moments, not results.
- **Cost measured on the gateway**: about 68k tokens per moment with GLM as tutor, 39k with a GPU tutor.
  Full run (100 moments × 2 prompts × 3 tutors): about 30M tokens, about 10 hours.

## Rules for this repo

- Never commit API keys: they live only in `.env`.
- No conversations or dataset records in Git or on Hugging Face (the dataset contains sessions with minors and
  its card asks not to redistribute records). Only ids, summaries and verdicts are published.
- Ai2's code is never edited: our changes are applied at runtime by `run_benchmark.py`.
- Every result states the benchmark commit and dataset revision it was produced with.

## Credits, licences and citation

This project builds on Ai2's work; nothing of theirs is redistributed by this repository.

- **Benchmark code**: [allenai/tutormoments](https://github.com/allenai/tutormoments), © 2026 The Allen Institute
  for Artificial Intelligence, Apache License 2.0. `tutormoments/` is a git **submodule**: a pointer to their
  repository at commit `c6e26d7`, not a copy. We do not modify it; our changes are applied at runtime by
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
