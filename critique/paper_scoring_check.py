"""Paper-intended scoring check: fixed weights from the paper/README, no tuning. Lookup on Person 2's candidate scores."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "critique"))
from learn_weights import features, select, first_step, POSITIONS, read_json
from bootstrap import load
OUT = Path(sys.argv[1])
items = {it["q_idx"]: it for it in read_json(REPO / "results/runs/asqa_cands_0_end.json")["data"]}
scores = {p: load(REPO / f"results/runs/critique_p2/candidate_{p}.json.perq.jsonl") for p in POSITIONS}
configs = {
    "released":      {"w_rel": 1, "w_sup": 1, "w_use": 0,   "use_seqscore": False},  # what the code really does (C1)
    "plus_utility":  {"w_rel": 1, "w_sup": 1, "w_use": 0.5, "use_seqscore": False},  # bug 1 fixed only
    "plus_seq":      {"w_rel": 1, "w_sup": 1, "w_use": 0,   "use_seqscore": True},   # bug 2 fixed only
    "paper_intended":{"w_rel": 1, "w_sup": 1, "w_use": 0.5, "use_seqscore": True},   # both: the paper's full score
}
for name, w in configs.items():
    rows, changed = [], 0
    for q, it in sorted(items.items()):
        path = it["best_paths"][str(it.get("output_path", 0))]
        orig = next(c["p_idx"] for c in first_step(it) if c["node_id"] == path[-1])
        p = select([(c, features(c, 1.0)) for c in first_step(it)], w)["p_idx"]
        changed += p != orig
        rows.append({**scores[p][q], "p_idx": p})
    (OUT / f"{name}.perq.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{name:15s} picks changed vs C1: {changed}/200")
