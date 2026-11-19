"""Build a balanced, reproducible subset of TutorMoments moments.

Picks N/2 scaffolding and N/2 rigor moments at random with a fixed seed from the
520 released moments, and writes them as a local release folder that the
benchmark can load with --data_path:

    <out>/moments.jsonl      the selected moments (same format as the release)
    <out>/subset.json        selection details: ids, seed, dataset revision, counts

Only subset.json is meant to be published: moments.jsonl contains the dataset
content and can be regenerated from the ids with this script.

Usage:
    python scripts/make_subset.py --n 100 --seed 0 --out subsets/subset_100
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tutormoments.moments import load_moments, records_content_hash

REPO = "allenai/tutormoments-preview"
REVISION = "3bb0c104d65e29facffca7ba864ef10fa22d0e78"

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100, help="total moments (half scaffolding, half rigor)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--revision", default=REVISION)
parser.add_argument("--data_path", default=None,
                    help="local folder with moments.jsonl (instead of downloading from HF)")
parser.add_argument("--out", default="subsets/subset_100")
args = parser.parse_args()

if args.data_path:
    moments, source = load_moments(data_path=args.data_path)
    print(f"loaded {len(moments)} moments from {args.data_path}")
else:
    moments, source = load_moments(dataset=REPO, revision=args.revision)
    print(f"loaded {len(moments)} moments from {REPO} @ {args.revision}")

# sort first so the result depends only on the seed, not on load order
by_dim = {"scaffolding": [], "rigor": []}
for m in sorted(moments, key=lambda m: m.id):
    by_dim[m.dimension].append(m)

rng = random.Random(args.seed)
half = args.n // 2
picked = rng.sample(by_dim["scaffolding"], half) + rng.sample(by_dim["rigor"], args.n - half)
picked.sort(key=lambda m: m.id)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
records = [m.to_dict() for m in picked]
with open(out / "moments.jsonl", "w", encoding="utf-8") as fh:
    for rec in records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# session = trailing id of the source transcript (same key as recompute_table8.py)
sessions = {m.provenance["conv_id"].rsplit("_", 1)[-1] for m in picked}
info = {
    "dataset": args.data_path or REPO,
    "revision": None if args.data_path else args.revision,
    "seed": args.seed,
    "n": len(picked),
    "by_dimension": dict(Counter(m.dimension for m in picked)),
    "n_sessions": len(sessions),
    "content_hash": records_content_hash(records),
    "ids": [m.id for m in picked],
}
with open(out / "subset.json", "w", encoding="utf-8") as fh:
    json.dump(info, fh, indent=2)

print(f"wrote {out}/moments.jsonl: {info['by_dimension']} from {info['n_sessions']} sessions")
print(f"wrote {out}/subset.json (ids + seed + revision)")
