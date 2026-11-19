"""Publish code and results to a Hugging Face dataset repository (private by default).

Step 1 builds a local folder (export/) with what can be shared; step 2 uploads it.
Look at export/ before uploading.

What goes in
  - docs/hf_card.md (as README.md, the dataset card), configs/, requirements.txt,
    scripts/*.py
  - subsets/*/subset.json            (moment ids + seed + dataset revision, no content)
  - results/<run>/summary.json and config.json
  - results/<run>/verdicts.jsonl     (per moment: id, gold, judge verdict, over-scaffold flag,
                                      the judge's action facets)
  - reports/comparison*.csv / .md / .png (if present)

What stays out
  - .env and any key
  - transcripts (they contain the dataset's real tutoring context, sessions with minors:
    the dataset card asks not to redistribute some records) -- share only after checking
    with the supervisor
  - logs/ (raw judge answers), subsets/*/moments.jsonl (dataset content)

Usage:
    python scripts/publish_hf.py --repo USER/socrate-tutormoments            # export only
    python scripts/publish_hf.py --repo USER/socrate-tutormoments --upload   # export + upload
    add --runs "*pilot_10*" to publish only some run folders
Needs `huggingface-cli login` (or HF_TOKEN in .env) for --upload.
"""

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="HF repo id, e.g. stefano/socrate-tutormoments")
parser.add_argument("--results", default="results")
parser.add_argument("--runs", default="*", help='which run folders to publish, e.g. "*pilot_10*"')
parser.add_argument("--subsets", default="subsets")
parser.add_argument("--export", default="export")
parser.add_argument("--upload", action="store_true", help="upload export/ to the HF repo")
parser.add_argument("--public", action="store_true", help="create the repo as public (default private)")
args = parser.parse_args()

out = Path(args.export)
if out.exists():
    shutil.rmtree(out)
(out / "scripts").mkdir(parents=True)

# dataset card (docs/hf_card.md becomes the README of the HF repo), code and configuration
shutil.copy(ROOT / "docs" / "hf_card.md", out / "README.md")
(out / "configs").mkdir()
for f in (ROOT / "configs").glob("*.yaml"):
    shutil.copy(f, out / "configs" / f.name)
for name in ["requirements.txt", ".gitignore"]:
    if (ROOT / name).exists():
        shutil.copy(ROOT / name, out / name)
for f in (ROOT / "scripts").glob("*.py"):
    shutil.copy(f, out / "scripts" / f.name)

# subsets: ids only
gold = {}
for info in Path(args.subsets).glob("*/subset.json"):
    dest = out / "subsets" / info.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(info, dest / "subset.json")
    moments_file = info.parent / "moments.jsonl"
    if moments_file.exists():
        for line in open(moments_file, encoding="utf-8"):
            m = json.loads(line)
            gold[m["id"]] = m["dimension"]

# results: summary, config and one verdict line per moment
n_runs = 0
for run in sorted(Path(args.results).glob(args.runs)):
    if not run.is_dir():
        continue
    if not (run / "summary.json").exists():
        continue
    dest = out / "results" / run.name
    dest.mkdir(parents=True)
    for name in ["summary.json", "config.json"]:
        shutil.copy(run / name, dest / name)
    with open(dest / "verdicts.jsonl", "w", encoding="utf-8") as fh:
        for f in sorted((run / "scores").glob("*.json")):
            sc = json.load(open(f, encoding="utf-8"))
            fh.write(json.dumps({
                "moment_id": sc["scenario_id"],
                "gold": gold.get(sc["scenario_id"]),
                "action_label": sc.get("action_label"),
                "overscaffold": bool(sc.get("overscaffold_decomposed")),
                "action_facets": sc.get("action_decomposed"),
                "judge_tokens": (sc.get("usage") or {}).get("total_tokens"),
            }, ensure_ascii=False) + "\n")
    n_runs += 1

for f in (ROOT / "reports").glob("comparison*"):
    if f.suffix in (".csv", ".md", ".png"):
        (out / "reports").mkdir(exist_ok=True)
        shutil.copy(f, out / "reports" / f.name)

# last check: nothing that looks like a secret or a transcript
for f in out.rglob("*"):
    if f.name == ".env" or "transcripts" in f.parts or f.name.startswith("judge_raw"):
        raise SystemExit(f"refusing to publish {f}")

print(f"export ready in {out}/ ({n_runs} runs)")

if args.upload:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset",
                      commit_message="Upload code and results")
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")
else:
    print("not uploaded: check the folder, then rerun with --upload")
