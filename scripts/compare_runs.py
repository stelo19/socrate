"""Compare tutors: S, R, O with 95% confidence intervals, one row per (tutor, prompt).

Reads every run folder in results/ (written by run_benchmark.py), takes the
judge's verdict for each moment from scores/*.json and computes the paper's
three metrics, with the same rules as tutormoments.report.aggregate:

  S = scaffolding moments where the tutor scaffolded (action "scaffolding" or
      "both") and the judge found no over-scaffolding, / scaffolding moments
  R = the same for rigor moments
  O = 1 - share of moments with over-scaffolding

Confidence intervals: bootstrap over sessions (moments from the same tutoring
session are not independent), as in recompute_table8.py.

Usage:
    python scripts/compare_runs.py --results results --data_path subsets/subset_100 \
        --out reports/comparison
writes reports/comparison.csv, .md (and .png if matplotlib is installed)
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results", help="folder with the run folders")
parser.add_argument("--data_path", required=True,
                    help="subset folder with moments.jsonl: gold dimension and session of each moment")
parser.add_argument("--n_boot", type=int, default=2000)
parser.add_argument("--out", default="reports/comparison")
args = parser.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)


def metrics(items):
    """items: list of (gold_dimension, action_label, has_overscaffold)."""
    scaf = [(a, o) for d, a, o in items if d == "scaffolding"]
    rig = [(a, o) for d, a, o in items if d == "rigor"]
    s = sum(a in ("scaffolding", "both") and not o for a, o in scaf) / len(scaf) if scaf else None
    r = sum(a in ("rigor", "both") and not o for a, o in rig) / len(rig) if rig else None
    o = 1 - sum(o for _, _, o in items) / len(items) if items else None
    return s, r, o


def bootstrap(by_session, n_boot, seed=0):
    """95% CI for (S, R, O): resample whole sessions with replacement."""
    rng = random.Random(seed)
    keys = list(by_session)
    draws = []
    for _ in range(n_boot):
        sample = []
        for _ in keys:
            sample.extend(by_session[rng.choice(keys)])
        draws.append(metrics(sample))
    cis = []
    for i in range(3):
        vals = sorted(d[i] for d in draws if d[i] is not None)
        if not vals:
            cis.append((None, None))
            continue
        cis.append((vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]))
    return cis


# moment id -> gold dimension and session (trailing id of the source transcript).
# The gold label comes from the dataset: the score file's annotation_type is the
# judge prompt used ("scaffolding" for every moment), not the gold label.
gold, sessions = {}, {}
with open(Path(args.data_path) / "moments.jsonl", encoding="utf-8") as fh:
    for line in fh:
        m = json.loads(line)
        gold[m["id"]] = m["dimension"]
        sessions[m["id"]] = m["provenance"]["conv_id"].rsplit("_", 1)[-1]

rows = []
for run in sorted(Path(args.results).iterdir()):
    if not (run / "config.json").exists() or not (run / "scores").is_dir():
        continue
    cfg = json.load(open(run / "config.json", encoding="utf-8"))
    by_session = defaultdict(list)
    tokens = 0
    for f in sorted((run / "scores").glob("*.json")):
        sc = json.load(open(f, encoding="utf-8"))
        mid = sc["scenario_id"]
        if mid not in gold:
            continue  # moment not in this subset
        item = (gold[mid], sc.get("action_label") or "neither",
                bool(sc.get("overscaffold_decomposed")))
        by_session[sessions[mid]].append(item)
        tokens += (sc.get("usage") or {}).get("total_tokens", 0)
    items = [x for v in by_session.values() for x in v]
    if not items:
        continue
    s, r, o = metrics(items)
    (s_lo, s_hi), (r_lo, r_hi), (o_lo, o_hi) = bootstrap(by_session, args.n_boot)
    rows.append({
        "run": run.name, "tutor": cfg["tutor"], "prompt": cfg["mode"],
        "student": (cfg.get("student") or {}).get("model"),
        "judge": (cfg.get("scorer") or {}).get("model"),
        "n": len(items), "n_sessions": len(by_session),
        "S": s, "S_lo": s_lo, "S_hi": s_hi,
        "R": r, "R_lo": r_lo, "R_hi": r_hi,
        "O": o, "O_lo": o_lo, "O_hi": o_hi,
        "judge_tokens": tokens,
    })

if not rows:
    raise SystemExit(f"no scored runs found in {args.results}")

rows.sort(key=lambda x: (x["prompt"], x["tutor"]))
with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def fmt(v, lo, hi):
    if v is None:
        return "-"
    return f"{100 * v:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


lines = ["| tutor | prompt | n | S (%) | R (%) | O (%) |", "|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['tutor']} | {x['prompt']} | {x['n']} | {fmt(x['S'], x['S_lo'], x['S_hi'])} "
                 f"| {fmt(x['R'], x['R_lo'], x['R_hi'])} | {fmt(x['O'], x['O_lo'], x['O_hi'])} |")
lines.append("")
lines.append(f"student: {rows[0]['student']} | judge: {rows[0]['judge']} | "
             f"95% CI: session bootstrap, {args.n_boot} resamples")
table = "\n".join(lines)
open(f"{args.out}.md", "w", encoding="utf-8").write(table + "\n")
print(table)
print(f"\nwrote {args.out}.csv and {args.out}.md")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: no chart")
else:
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.6 * len(rows) + 1.5), sharey=True)
    labels = [f"{x['tutor']} ({x['prompt']})" for x in rows]
    for ax, key, title in zip(axes, "SRO", ["Appropriate scaffolding", "Appropriate rigor",
                                             "Avoids over-scaffolding"]):
        for i, x in enumerate(rows):
            if x[key] is None:
                continue
            ax.errorbar(100 * x[key], i, fmt="o", color="#3b6ea8",
                        xerr=[[100 * (x[key] - x[key + "_lo"])], [100 * (x[key + "_hi"] - x[key])]])
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.png")
