"""Make the two fixed ASQA test sets (Set A and Set B) used by every division for quick checks.

Picks 200 different questions out of the 948 ASQA dev questions with a fixed random seed, then gives the
first 100 to Set A and the next 100 to Set B. Indices are positions in
data/eval_data/asqa_eval_gtr_top100.json (0-based), which is also the order of every output file.

Run once (no GPU, no data needed), then commit the files it writes:
    python critique/make_splits.py
Optional check that the indices point at real questions (needs the data file):
    python critique/make_splits.py --data data/eval_data/asqa_eval_gtr_top100.json
"""
import argparse
import json
import os
import random

N_TOTAL = 948
SEED = 6103  # fixed: never change it once the files are committed
SIZE = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--data", default=None, help="optional: ASQA eval file, to record each question's text")
    args = parser.parse_args()

    picked = random.Random(SEED).sample(range(N_TOTAL), 2 * SIZE)
    splits = {"A": sorted(picked[:SIZE]), "B": sorted(picked[SIZE:])}

    questions = None
    if args.data:
        data = json.load(open(args.data))
        assert len(data) == N_TOTAL, f"expected {N_TOTAL} questions, found {len(data)}"
        questions = [item["question"] for item in data]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, indices in splits.items():
        record = {"name": name, "seed": SEED, "n_total": N_TOTAL, "indices": indices}
        if questions is not None:
            record["questions"] = [questions[i] for i in indices]
        path = os.path.join(args.out_dir, f"asqa_{name}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=1)
        print(f"wrote {path}: {len(indices)} questions")
    assert not set(splits["A"]) & set(splits["B"]), "Set A and Set B overlap"


if __name__ == "__main__":
    main()
