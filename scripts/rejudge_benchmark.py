"""Re-judge the paper's published conversations with another judge model (stress test).

What it does
  1. Loads the 520 moments and the paper's 7,280 scored conversations
     (benchmark_520) at the pinned dataset revision.
  2. Picks the conversations of one tutor model and one prompt mode, balanced
     between scaffolding and rigor moments (first N/2 of each, in dataset order).
  3. Sends them to the judge set in the config (e.g. glm-5.3-flash on the EURECOM
     gateway), using the benchmark's own scorer: scoring.score_batch, with the
     same three passes and prompts as the paper. Nothing is regenerated: the
     tutor and student turns are exactly those published by Ai2.
  4. Compares the new verdicts with the paper's judge (Claude Opus 4.6):
     agreement on the action label and on over-scaffolding, and S, R, O
     computed on the same moments with both judges.
  5. Saves one CSV row per conversation.

Usage (from the repo root, venv active, EURECOM_API_KEY in .env):
    python scripts/rejudge_benchmark.py --config configs/socrate_config.yaml \
        --tutor gpt-5.5-2026-04-23 --mode plain --n 10
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

REPO = "allenai/tutormoments-preview"
REVISION = "3bb0c104d65e29facffca7ba864ef10fa22d0e78"

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="config whose scorer is the new judge")
parser.add_argument("--tutor", default="gpt-5.5-2026-04-23", help="tutor_model in benchmark_520")
parser.add_argument("--mode", default="plain", help="plain or scaffolding_rigor")
parser.add_argument("--n", type=int, default=10, help="number of conversations (half scaffolding, half rigor)")
parser.add_argument("--revision", default=REVISION)
parser.add_argument("--data_path", default=None, help="local moments (testing only)")
parser.add_argument("--out", default=None, help="CSV path (default: results/rejudge_<tutor>_<mode>_<n>.csv)")
args = parser.parse_args()

# the benchmark reads its config from this variable: the new judge is the config's scorer
os.environ["TUTORMOMENTS_CONFIG"] = os.path.abspath(args.config)

# importing run_benchmark activates the routing of the judge (glm-...) to the gateway
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmark  # noqa: E402

from datasets import load_dataset  # noqa: E402
from tutormoments.config import scorer_spec  # noqa: E402
from tutormoments.conversation import Transcript  # noqa: E402
from tutormoments.moments import load_moments  # noqa: E402
from tutormoments.report import aggregate  # noqa: E402
from tutormoments.scoring import score_batch  # noqa: E402

judge = scorer_spec().model
if run_benchmark.is_gateway_model(judge) and not run_benchmark.GATEWAY.get("api_key"):
    raise SystemExit("set EURECOM_API_KEY in .env to use the gateway judge")
print(f"new judge: {judge} | tutor: {args.tutor} | mode: {args.mode} | n: {args.n}")

# 1. data
if args.data_path:
    moments, _ = load_moments(data_path=args.data_path)
else:
    moments, _ = load_moments(dataset=REPO, revision=args.revision)
moment_by_sid = {m.id.split(":", 1)[-1]: m for m in moments}  # moments: "balanced_520:<sid>"
bench = load_dataset(REPO, "benchmark_520", revision=args.revision)["train"]

# 2. pick the conversations: one tutor, one mode, balanced scaffolding / rigor
rows = [r for r in bench if r["tutor_model"] == args.tutor and r["prompt_mode"] == args.mode]
if not rows:
    tutors = sorted({r["tutor_model"] for r in bench})
    raise SystemExit(f"no rows for {args.tutor}/{args.mode}. Tutors available: {tutors}")
picked, quota = [], {"scaffolding": args.n // 2, "rigor": args.n - args.n // 2}
for r in rows:
    m = moment_by_sid.get(r["scenario_id"])
    if m is not None and quota.get(m.dimension, 0) > 0:
        picked.append((m, r))
        quota[m.dimension] -= 1

# 3. judge them again with the benchmark's own scorer
pairs = []
for m, r in picked:
    turns = [dict(t) for t in r["exchange"]["generated_turns"]]
    # the scorer needs a turn_number on every generated turn: number them after the
    # last real turn if the published rows do not carry it
    last = max((c.get("turn_number", 0) or 0) for c in m.context) if m.context else 0
    for i, turn in enumerate(turns, 1):
        if turn.get("turn_number") is None:
            turn["turn_number"] = last + i
    t = Transcript(scenario_id=m.id, tutor_model=r["tutor_model"], generated_turns=turns)
    pairs.append((m, t))
new = score_batch(pairs)

# 4. compare with the paper's judge
def paper_verdict(row):
    anns = (row.get("annotation") or {}).get("annotations") or []
    return anns[0] if anns else {}


out_rows, old_anns, new_anns, scen = [], [], [], []
for m, r in picked:
    old = paper_verdict(r)
    nw = new[m.id].to_dict()
    old_over = bool(old.get("overscaffold_decomposed"))
    new_over = bool(nw.get("overscaffold_decomposed"))
    out_rows.append({
        "scenario_id": m.id, "gold": m.dimension,
        "paper_label": old.get("action_label"), "new_label": nw.get("action_label"),
        "paper_overscaffold": old_over, "new_overscaffold": new_over,
        "label_agree": old.get("action_label") == nw.get("action_label"),
        "over_agree": old_over == new_over,
        "new_action": nw.get("action"),
    })
    scen.append(SimpleNamespace(dimension=m.dimension))
    old_anns.append(SimpleNamespace(action_label=old.get("action_label"),
                                    overscaffold_decomposed=list(old.get("overscaffold_decomposed") or [])))
    new_anns.append(SimpleNamespace(action_label=nw.get("action_label"),
                                    overscaffold_decomposed=list(nw.get("overscaffold_decomposed") or [])))


def sro(anns):
    a = aggregate(scen, anns)
    rate = a["overscaffold"]["rate"]
    return a["scaffold_calibrated"]["score"], a["rigor_calibrated"]["score"], None if rate is None else 1 - rate


def f(x):
    return "  -  " if x is None else f"{x:.3f}"


n = len(out_rows)
print(f"\nconversations re-judged: {n}")
print(f"action label agreement:   {sum(r['label_agree'] for r in out_rows)}/{n}")
print(f"over-scaffolding agreement: {sum(r['over_agree'] for r in out_rows)}/{n}")
print(f"\n{'judge':<26} {'S':>6} {'R':>6} {'O':>6}")
for name, anns in [("paper (claude-opus-4-6)", old_anns), (f"new ({judge})", new_anns)]:
    s, r_, o = sro(anns)
    print(f"{name:<26} {f(s):>6} {f(r_):>6} {f(o):>6}")
print("\nper conversation:")
for r in out_rows:
    print(f"  {r['gold']:<11} paper={str(r['paper_label']):<11} new={str(r['new_label']):<11} "
          f"over paper/new={int(r['paper_overscaffold'])}/{int(r['new_overscaffold'])}")

path = args.out or f"results/rejudge_{args.tutor}_{args.mode}_{n}.csv"
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nsaved {path}")
