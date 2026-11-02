"""Recompute Table 8 of the TutorMoments working paper from the released data.
 
No API key and no cost: it only reads the replays and judge annotations that
Ai2 published in the `benchmark_520` config of allenai/tutormoments-preview.
 
What it does
  1. Loads `moments` (gold label per moment) and `benchmark_520` (7,280 scored
     replays = 520 moments x 7 tutor models x 2 prompt modes) at a pinned revision.
  2. For each (tutor model, prompt mode) cell, computes the paper's three metrics
     with the repository's own function `tutormoments.report.aggregate`:
       Appropriate Scaffolding, Appropriate Rigor, Avoids Over-scaffolding.
  3. Does it on two populations:
       - "paper"  : rows with source_round in {s42, s43, topup}
                    (the repo says these are the paper's ~100-moment cells)
       - "all520" : all 520 moments per cell
  4. Adds 95% confidence intervals with a bootstrap that resamples whole
     tutoring sessions (moments from the same session are not independent).
  5. Prints both against the paper's published values and saves a CSV.
 
Usage (inside the repo's virtual environment):
    python recompute_table8.py
    python recompute_table8.py --revision <sha> --boot 2000 --out table8_recomputed.csv
"""
 
from __future__ import annotations
 
import argparse
import csv
import random
from collections import Counter, defaultdict
from types import SimpleNamespace
 
from datasets import load_dataset
from tutormoments.report import aggregate
 
REPO = "allenai/tutormoments-preview"
DEFAULT_REVISION = "3bb0c104d65e29facffca7ba864ef10fa22d0e78"  # pinned 2 Nov 2026
PAPER_ROUNDS = {"s42", "s43", "topup"}
 
# Paper Table 8: (S, R, O) for plain and evaluation-aware (= scaffolding_rigor)
PAPER = {
    "Claude Opus 4.8":   {"plain": (0.615, 0.208, 0.462), "scaffolding_rigor": (0.858, 0.831, 0.896)},
    "Claude Sonnet 4.6": {"plain": (0.573, 0.085, 0.437), "scaffolding_rigor": (0.831, 0.792, 0.913)},
    "DeepSeek V4 Pro":   {"plain": (0.485, 0.088, 0.379), "scaffolding_rigor": (0.712, 0.492, 0.623)},
    "Gemini 2.5 Pro":    {"plain": (0.727, 0.223, 0.546), "scaffolding_rigor": (0.896, 0.712, 0.854)},
    "Gemini 3.5 Flash":  {"plain": (0.723, 0.158, 0.598), "scaffolding_rigor": (0.885, 0.646, 0.817)},
    "GPT 5.5":           {"plain": (0.269, 0.046, 0.173), "scaffolding_rigor": (0.773, 0.531, 0.665)},
    "GPT 5.4 mini":      {"plain": (0.181, 0.023, 0.188), "scaffolding_rigor": (0.738, 0.404, 0.598)},
}
 
 
def paper_name(tutor_model: str) -> str | None:
    """Map a dataset tutor_model id (e.g. 'gpt-5.5-2026-04-23') to the paper's row name."""
    t = tutor_model.lower().replace("_", "-")
    rules = [
        ("opus-4-8", "Claude Opus 4.8"), ("opus-4.8", "Claude Opus 4.8"),
        ("sonnet-4-6", "Claude Sonnet 4.6"), ("sonnet-4.6", "Claude Sonnet 4.6"),
        ("deepseek", "DeepSeek V4 Pro"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"), ("gemini-3.5-flash", "Gemini 3.5 Flash"),
        ("gpt-5.4-mini", "GPT 5.4 mini"), ("gpt-5.5", "GPT 5.5"),
    ]
    for key, name in rules:
        if key in t:
            return name
    return None
 
 
def first_annotation(row: dict) -> dict | None:
    """The judge's output for one replay (None if scoring failed for this row)."""
    anns = (row.get("annotation") or {}).get("annotations") or []
    return anns[0] if anns else None
 
 
def metrics(items: list[tuple[str, dict]]) -> tuple[float | None, float | None, float | None]:
    """(S, R, O) for a list of (gold dimension, annotation) pairs, via the repo's own rule."""
    if not items:
        return (None, None, None)
    scen = [SimpleNamespace(dimension=d) for d, _ in items]
    anns = [
        SimpleNamespace(
            action_label=a.get("action_label"),
            overscaffold_decomposed=list(a.get("overscaffold_decomposed") or []),
        )
        for _, a in items
    ]
    agg = aggregate(scen, anns)
    s = agg["scaffold_calibrated"]["score"]
    r = agg["rigor_calibrated"]["score"]
    rate = agg["overscaffold"]["rate"]
    o = None if rate is None else 1.0 - rate
    return (s, r, o)
 
 
def cluster_bootstrap(items_by_session: dict[str, list], n_boot: int, seed: int = 0):
    """95% CI for (S, R, O): resample whole sessions with replacement."""
    rng = random.Random(seed)
    sessions = list(items_by_session)
    draws = []
    for _ in range(n_boot):
        sample = []
        for _ in sessions:
            sample.extend(items_by_session[rng.choice(sessions)])
        draws.append(metrics(sample))
    cis = []
    for k in range(3):
        vals = sorted(d[k] for d in draws if d[k] is not None)
        if not vals:
            cis.append((None, None))
            continue
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        cis.append((lo, hi))
    return cis
 
 
def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (0 = no CIs)")
    ap.add_argument("--out", default="table8_recomputed.csv")
    args = ap.parse_args()
 
    print(f"Dataset {REPO} @ {args.revision}\n")
    moments = load_dataset(REPO, "moments", revision=args.revision)["train"]
    bench = load_dataset(REPO, "benchmark_520", revision=args.revision)["train"]
 
    # Gold label per moment. moments.id = "balanced_520:<scenario_id>"
    gold = {m["id"].split(":", 1)[-1]: m["dimension"] for m in moments}
    print("Gold labels in `moments`:", dict(Counter(gold.values())))
 
    # Sanity checks on the replay table
    print("Rows per source_round:", dict(Counter(bench["source_round"])))
    unmapped = sorted({t for t in bench["tutor_model"] if paper_name(t) is None})
    if unmapped:
        print("WARNING: tutor models not mapped to a paper row:", unmapped)
 
    # Collect (dimension, annotation) per cell and population, grouped by session
    data = {"paper": defaultdict(lambda: defaultdict(list)),
            "all520": defaultdict(lambda: defaultdict(list))}
    missing_gold, failed = 0, []
    round_counts = defaultdict(Counter)
    for row in bench:
        cell = (row["tutor_model"], row["prompt_mode"])
        dim = gold.get(row["scenario_id"])
        if dim is None:
            missing_gold += 1
            continue
        ann = first_annotation(row)
        if ann is None:
            failed.append((cell, row["scenario_id"], row["source_round"]))
            continue
        session = row["conv_id"].rsplit("_", 1)[-1]  # trailing UUID = transcript id
        data["all520"][cell][session].append((dim, ann))
        if row["source_round"] in PAPER_ROUNDS:
            data["paper"][cell][session].append((dim, ann))
            round_counts[cell][dim] += 1
 
    print(f"Rows with no gold label: {missing_gold}")
    print(f"Rows with failed scoring (excluded): {len(failed)} {failed[:5]}")
    print("\nPaper subset size per cell (scaffolding / rigor):")
    for cell in sorted(round_counts):
        c = round_counts[cell]
        print(f"  {cell[0]:<32} {cell[1]:<18} {c['scaffolding']:>4} / {c['rigor']:<4}")
 
    # Compute and print
    rows_out = []
    header = (f"\n{'Model':<18} {'Prompt':<10} {'Pop':<7} {'n':>4} | "
              f"{'S':>6} {'S paper':>8} | {'R':>6} {'R paper':>8} | {'O':>6} {'O paper':>8}")
    print(header)
    print("-" * len(header))
    cells = sorted(set(data["all520"]) | set(data["paper"]),
                   key=lambda c: (paper_name(c[0]) or c[0], c[1]))
    for cell in cells:
        name = paper_name(cell[0]) or cell[0]
        ref = PAPER.get(name, {}).get(cell[1], (None, None, None))
        prompt = "plain" if cell[1] == "plain" else "eval-aw."
        for pop in ("paper", "all520"):
            by_session = data[pop].get(cell, {})
            items = [x for v in by_session.values() for x in v]
            s, r, o = metrics(items)
            cis = cluster_bootstrap(by_session, args.boot) if (args.boot and items) else [(None, None)] * 3
            print(f"{name:<18} {prompt:<10} {pop:<7} {len(items):>4} | "
                  f"{fmt(s):>6} {fmt(ref[0]):>8} | {fmt(r):>6} {fmt(ref[1]):>8} | "
                  f"{fmt(o):>6} {fmt(ref[2]):>8}")
            rows_out.append({
                "model": name, "tutor_model_id": cell[0], "prompt_mode": cell[1],
                "population": pop, "n_moments": len(items),
                "n_sessions": len(by_session),
                "S": s, "S_ci_low": cis[0][0], "S_ci_high": cis[0][1], "S_paper": ref[0],
                "R": r, "R_ci_low": cis[1][0], "R_ci_high": cis[1][1], "R_paper": ref[1],
                "O": o, "O_ci_low": cis[2][0], "O_ci_high": cis[2][1], "O_paper": ref[2],
            })
 
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved {len(rows_out)} rows (with 95% session-bootstrap CIs) to {args.out}")
 
 
if __name__ == "__main__":
    main()

