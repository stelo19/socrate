# TutorMoments — Local Setup and Reproduction Guide

EURECOM semester project · Stefano Longo, Daniele Priola · 21 Oct 2026

This guide gets the TutorMoments benchmark running on your laptops, from an empty machine to a verified run. Every command in Parts B–D was **executed and checked** on a clean Linux machine on 21 Oct 2026 (repo commit `c6e26d7`, Python 3.13). Parts E–G depend on your API access and the Hugging Face download, which I could not run from my side; they are marked accordingly.

---

## Time budget

| Part | What | Needs | Your time | Notes |
|---|---|---|---|---|
| A | Prerequisites (Python ≥ 3.11, Git) | — | 10–20 min (Windows: +30–45 min for WSL) | Once per laptop |
| B | Clone + virtual environment + install | Internet | 5–10 min | Install took 76 s here |
| C | Run the test suite | Nothing | 2 min | 766 tests in ~10 s, all API calls mocked |
| D | Keyless end-to-end replay with dummy tutor/student | Nothing | 10–15 min | Proves the replay loop works on your machine |
| E | Download the dataset, pin its revision | Internet (~150 MB) | 10–20 min | Send me the output |
| F | Recompute the paper's Table 8 offline | Part E output | 1–2 h | I write the script from your Part E output |
| G | Small real run (optional) | Anthropic API key + budget | 30 min of work, up to a few hours of waiting | ~$2–8 estimated; ask the supervisor first |

**Free path (A → F): about half a day for one person, mostly in one sitting.** Do A–D on *both* laptops; E–F on one is enough. G only after the supervisor agrees on a budget.

---

## Part A — Prerequisites

### A.1 Which operating system?

| OS | Recommendation |
|---|---|
| macOS | Native terminal. Fine. |
| Linux | Native terminal. Fine. |
| **Windows** | **Use WSL2 (Ubuntu), not native Windows.** |

**Why WSL on Windows.** The benchmark names result files after the moment id, which contains a colon (for example `transcripts/balanced_520:<id>.json`; verified in the dummy run of Part D). Windows does not allow `:` in file names, so a native Windows run will very likely fail when writing results. WSL gives you a real Linux environment inside Windows.

**Install WSL (Windows only):**
1. Open **PowerShell as Administrator** (Start menu → type "PowerShell" → right-click → Run as administrator).
2. Run:
   ```powershell
   wsl --install -d Ubuntu
   ```
   *What it does:* enables WSL2 and installs Ubuntu Linux.
3. Restart the computer when asked. Ubuntu opens and asks you to create a Linux username and password (this password is only for Linux `sudo`).
4. From now on, **every command in this guide runs inside the Ubuntu terminal**, not PowerShell. Work in your Linux home folder (`cd ~`), not under `/mnt/c/...`, which is much slower.

### A.2 Check Python and Git

Open a terminal (macOS: Terminal app; Linux/WSL: Ubuntu terminal) and run:

```bash
python3 --version
git --version
```

**Expected:** `Python 3.11.x` or higher, and any Git version.

**If Python is missing or older than 3.11:**

| System | Command |
|---|---|
| macOS | Install from python.org (3.12) or `brew install python@3.12` |
| Ubuntu 24.04 / WSL Ubuntu (default) | `sudo apt update && sudo apt install -y python3 python3-venv python3-pip git` (ships 3.12) |
| Ubuntu 22.04 | Ships 3.10, too old. Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh \| sh`), then use `uv venv --python 3.12 .venv` instead of the `python3 -m venv` step in B.3 |

**If Git is missing:** macOS `xcode-select --install`; Ubuntu `sudo apt install -y git`.

---

## Part B — Get the code and install it

### B.1 Choose a folder and clone

```bash
cd ~
mkdir -p eurecom && cd eurecom
git clone https://github.com/allenai/tutormoments.git
cd tutormoments
```

*What it does:* downloads the repository into `~/eurecom/tutormoments` and moves you into it.
**Check:** `ls` shows `README.md`, `pyproject.toml`, `src`, `tests`, `tutormoments_build`, `analysis`, `docs`.

### B.2 Record (and align) the code version

```bash
git log -1 --format='%h %ad %s' --date=short
```

*Why:* the repo is still changing (several commits on 19 Oct). Both of you should use the **same commit**, and every result you report must state it.
To use exactly the commit verified in this guide:

```bash
git checkout c6e26d7
```

(This puts Git in "detached HEAD" state — normal, nothing to fix.)

### B.3 Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

*What it does:* creates an isolated Python installation in `.venv/` so the project's packages don't touch your system Python; `activate` switches your terminal to it.
**Check:** your prompt now starts with `(.venv)`.
**Remember:** in every new terminal, run `cd ~/eurecom/tutormoments && source .venv/bin/activate` again.

### B.4 Install the package

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
```

*What it does:* installs TutorMoments in "editable" mode (`-e`: code changes take effect immediately) plus `pytest` (`dev`) and `pandas`/`matplotlib`/`seaborn` (`analysis`). Dependencies come from `pyproject.toml`; don't install other versions by hand.
**Expected:** ends with `Successfully installed ...` — about 1–2 minutes.
**Check:**

```bash
tutormoments --help
```

**Expected output (first lines):**
```text
usage: tutormoments [-h] {run,latency,report,view,taxonomy,smoke} ...
Tutormoments benchmark runner
```

---

## Part C — Run the test suite (no API keys, no cost)

```bash
pytest tests/tutormoments -q
```

*What it does:* runs the runtime's unit tests. Every provider SDK (Anthropic, OpenAI, Google…) is **mocked**, so no network call and no cost.
**Expected (commit `c6e26d7`):**
```text
766 passed, 13 skipped in ~8s
```
Skipped tests are normal (they need optional extras). Any `failed` line is a problem — copy the full output and send it to me.

---

## Part D — Keyless end-to-end replay with a dummy tutor and student

**Goal:** prove that the real benchmark loop (load moment → replay 5 turns → write transcript) runs on your machine, without any API key. The final scoring step *will* fail, on purpose: it needs Anthropic. That failure is the expected result.

### D.1 Create a scratch folder (outside the repo)

```bash
mkdir -p ~/eurecom/keyless && cd ~/eurecom/keyless
```

### D.2 Create `wrapper.py`

Create a file `wrapper.py` in that folder with this content:

```python
from tutormoments import register_tutor, register_student
from tutormoments.cli import main

@register_tutor("echo-tutor")
def echo_tutor(conversation):
    # A fake tutor: always asks for an explanation (a "rigor" move)
    return "Can you explain how you got that answer?"

@register_student("echo-student")
def echo_student(conversation):
    # A fake student: always gives the same reply
    return "I added the two numbers together."

if __name__ == "__main__":
    main()
```

*What it does:* registers two fake models through the repo's official plug-in mechanism (`register_tutor`, `register_student`, documented in the README) and then starts the normal command line.

### D.3 Create a config that uses the fake student

The simulated student is set in the config, not on the command line. A config file **replaces** the default entirely (it does not merge), so copy the full default and change one line:

```bash
sed 's/^student: .*/student: { model: echo-student, mode: oracle }/' \
  ~/eurecom/tutormoments/src/tutormoments/default_config.yaml > cfg.yaml
grep '^student' cfg.yaml
```

**Expected:** `student: { model: echo-student, mode: oracle }`

### D.4 Run on the repo's tiny test dataset

```bash
python wrapper.py run --config cfg.yaml --tutors echo-tutor --modes plain \
  --sample 1 --data_path ~/eurecom/tutormoments/tests/tutormoments/fixtures/mini_release
```

**Expected:** the command ends with an error:
```text
RuntimeError: No scenarios completed for echo-tutor/plain trial 1; 1 of 1 attempted scenarios failed.
```
That is correct. Now look at the log:

```bash
grep -E 'replay OK|Classification' results/*/run.log
```

**Expected:**
```text
... [trial 1][1/1] replay OK: mini:conv1__hum_1_2
... Starting Classification (trial 1/1): 1 replays, 3 pooled batch passes
... Classification failed for all pooled replays: ANTHROPIC_API_KEY not found in environment
```

**What this proves:** loading, replay and transcript writing work; only scoring needs a key.
**Inspect the output:**

```bash
ls -R results
python -m json.tool results/*/transcripts/*.json | head -60
```

You'll see the moment's pre-cut context followed by the 5 generated turns (tutor, student, tutor, student, tutor). `config.json` records the resolved config, git commit, and prompt hashes — this is how a run documents itself.

---

## Part E — Download the dataset and pin its version

Go back to the repo environment (any folder is fine, the venv must be active) and save this as `inspect_dataset.py`:

```python
from huggingface_hub import HfApi
from datasets import get_dataset_config_names, load_dataset

REPO = "allenai/tutormoments-preview"
sha = HfApi().dataset_info(REPO).sha          # exact dataset version
print("revision:", sha)

for cfg in get_dataset_config_names(REPO, revision=sha):
    ds = load_dataset(REPO, cfg, revision=sha)
    for split, d in ds.items():
        print(cfg, split, d.num_rows, d.column_names)
```

Run it:

```bash
python inspect_dataset.py | tee dataset_inventory.txt
```

*What it does:* reads the current dataset commit (`sha`), downloads every subset at exactly that commit (~150 MB, cached under `~/.cache/huggingface/`), prints row counts and columns, and saves the output to `dataset_inventory.txt`.
**Expected:** a `revision:` line with a 40-character hash, then one line per subset (`transcripts`, `annotations`, `ground_truth`, `moments`, `benchmark_520`, and probably `action_taxonomy`).
**Send me `dataset_inventory.txt`.** From it I write the Part F script with the real column names.

*Not verified on my side:* the Hugging Face Hub is blocked from my sandbox, so I haven't seen this output myself.

---

## Part F — Recompute the paper's Table 8 offline (free)

I'll provide `recompute_table8.py` once I have your Part E output. It will:
1. Load `benchmark_520` at the pinned revision.
2. Compute Appropriate Scaffolding, Appropriate Rigor and Avoids Over-scaffolding for each of the 14 cells (7 models × 2 prompts), using the formulas in paper §4.4.
3. Do it twice: on all 520 moments, and on the paper's subset (`source_round` ∈ {s42, s43, topup}, 100 moments per cell according to the repo's build code).
4. Print a comparison table against the paper's values, with confidence intervals.

This is the core result to show at the meeting: a replication of the paper's numbers from the released replays and judgments, with no API cost.

---

## Part G — Small real run (optional, costs money)

**Do this only after the supervisor agrees on a budget.**

### G.1 What you need
- An **Anthropic API key** with credits, from the Anthropic Console. A claude.ai subscription does not include API credits. This key is needed **whatever tutor you test**, because the simulated student and the judge are both Claude Opus 4.6 (paper Table 14, `default_config.yaml`).
- Using a **Claude tutor** (e.g. `claude-sonnet-4-6`, one of the paper's models) means one key is enough. An OpenAI/Google tutor needs that provider's key too.

### G.2 Store the key safely

In the repo root (`~/eurecom/tutormoments`), create a file named `.env`:

```text
ANTHROPIC_API_KEY=paste-your-key-here
```

- The code loads it automatically (`load_dotenv(override=True)` in `src/tutormoments/client.py`). `override=True` means the `.env` value wins over any key already set in your shell.
- `.env` is already in `.gitignore`, so Git won't commit it. Still: never paste the key into chat, screenshots, code files or shared docs.
- Run `tutormoments` commands from the repo root.

### G.3 Live connection check (costs cents)

```bash
tutormoments smoke --arms claude-sonnet-4-6 --providers anthropic
```

*What it does:* one tiny real call per role (tutor arm, student, scorer) plus one submit-then-cancel batch, to confirm your key and the request formats work.
**Expected:** exit without errors; exit code 0 (`echo $?` prints `0`). Code 2 means a configuration problem such as a missing key.

### G.4 Ten-moment run

```bash
tutormoments run --tutors claude-sonnet-4-6 --modes plain scaffolding_rigor \
  --sample 10 --dataset-revision <SHA from Part E> --log-file run_sample10.log
```

- 10 moments × 2 prompts = 20 replays, each 3 tutor + 2 student calls, then 3 scoring passes and the taxonomy classification.
- **Waiting time:** replays take a few minutes. Scoring goes through Anthropic's **Batch API**, which is cheaper but asynchronous: usually minutes, sometimes longer. The config allows up to 24 h. The command waits on its own; keep the terminal open. If interrupted, re-run the same command: it skips moments already done.
- **Cost (my rough estimate, not from the repo):** about $0.10–0.40 per replay including student and scoring, so roughly **$2–8** for this run. The exact cost is written into `summary.json` at the end.
- **Caveat:** `--sample 10` takes the *first* 10 moments, not a balanced sample. Scores on 10 moments are noise — this is a pipeline check, not a replication.

### G.5 Look at the outputs

```text
results/<run_id>/
  config.json          resolved config, git commit, dataset revision, prompt hashes
  run.log              what happened, including any failed moments
  transcripts/*.json   the generated replays
  scores/*.json        the judge's annotations per moment
  summary.json         metrics, failures, token usage, cost
  taxonomy/            per-action A–M classification (classified.csv)
```

Build the leaderboard and an HTML viewer:

```bash
tutormoments report --results-root results --out leaderboard
tutormoments view --results-root results --out viewer.html
```

`leaderboard/leaderboard.md` has the three paper metrics, KL values and cost; the first row is the human reference. Open `viewer.html` in a browser (WSL: `explorer.exe viewer.html`).

**Rate-limit errors (HTTP 429)** on new API accounts: add `--concurrency 1`.

---

## What to bring to the meeting

- [ ] Both laptops: Parts B–C done, `766 passed` (or the exact result if it differs).
- [ ] Part D log showing `replay OK` and the generated transcript.
- [ ] Recorded versions: repo commit, dataset revision SHA, Python version.
- [ ] `dataset_inventory.txt`.
- [ ] Part F: recomputed Table 8 vs the paper's values.
- [ ] (If approved) Part G: `leaderboard.md` and the run cost from `summary.json`.
- [ ] Questions for the supervisor: API budget; 100-moment paper cells vs the full 520; which stress test to focus on.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `tutormoments: command not found` | Virtual environment not active | `cd ~/eurecom/tutormoments && source .venv/bin/activate` |
| `requires a different Python: 3.10` during install | Python too old | See A.2 (install 3.12, or use `uv venv --python 3.12`) |
| `KeyError: 'defaults'` | Your `--config` file only contained the changed lines | Copy the full default config, as in D.3 |
| `ANTHROPIC_API_KEY not found in environment` | No key, `.env` in the wrong folder, or a typo | Expected in Part D. In Part G, put `.env` in the repo root and check the variable name |
| Error writing a file whose name contains `:` | Running on native Windows | Use WSL (A.1) |
| Hugging Face download very slow or rate-limited | Anonymous access limits | Retry; optionally log in with `huggingface-cli login` (free account) |
| HTTP 429 during a run | API rate limit | `--concurrency 1`, then re-run (finished moments are skipped) |
| Run seems stuck after replays | Waiting for batch scoring | Normal; check `run.log`; don't kill it — re-running resumes anyway |
| Any test fails in Part C | Version mismatch or broken install | Send me the full `pytest` output and `git log -1` |
