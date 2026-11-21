"""How often does DeBERTa fail to entail an answer sentence that is (almost) copied from its own passage? (CPU)

A sentence counts as near-verbatim when >= 90% of its word tokens appear, in order, as one contiguous run in the
passage (after lower-casing and dropping punctuation), i.e. longest common token substring / sentence length.
Prints the p_ent distribution for near-verbatim sentences and writes the failures (p_ent < 0.5) to a JSONL that
grounding/nli_probe.py (GPU) re-scores under controlled variations.

    python grounding/verbatim_check.py --out results/runs/verbatim_fail_AB.jsonl
"""
import argparse
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
W = re.compile(r"[\w]+", re.U)


def toks(s):
    return W.findall(s.lower())


def lcs_run(a, b):
    """Longest common contiguous token run between lists a and b."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(ROOT, "results/runs/asqa_cands_0_end.json"))
    ap.add_argument("--ground", default=",".join(os.path.join(ROOT, f"results/runs/cand_ground_{s}.jsonl") for s in "AB"))
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/runs/verbatim_fail_AB.jsonl"))
    a = ap.parse_args()

    ctx = {}
    for it in json.load(open(a.cands, encoding="utf-8"))["data"]:
        for c in it["candidates"]:
            if int(c["depth"]) == 1:
                ctx[(it["q_idx"], int(c["node_id"]))] = c["ctx"]
    rows = []
    for path in a.ground.split(","):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["sent_idx"] < 0:
                continue
            c = ctx[(r["q_idx"], int(r["node_id"]))]
            st, pt = toks(r["sentence"]), toks(c["title"] + " " + c["text"])
            if len(st) < 4:
                continue
            r["overlap"] = lcs_run(st, pt) / len(st)
            r["title"], r["text"] = c["title"], c["text"]
            rows.append(r)
    vb = [r for r in rows if r["overlap"] >= a.thr]
    fail = [r for r in vb if r["p_ent"] < 0.5]
    print(f"sentences: {len(rows)}; near-verbatim (>= {a.thr:.0%} contiguous): {len(vb)}")
    print(f"near-verbatim with p_ent < 0.5: {len(fail)} ({len(fail) / max(1, len(vb)):.0%}); "
          f"median p_ent {statistics.median(r['p_ent'] for r in vb):.3f}")
    print("  of the failures: neutral-dominant", sum(r["p_neu"] > r["p_con"] for r in fail),
          "contradiction-dominant", sum(r["p_con"] >= r["p_neu"] for r in fail))
    exact = [r for r in vb if r["overlap"] == 1.0]
    print(f"exact copies: {len(exact)}, of which p_ent < 0.5: {sum(r['p_ent'] < 0.5 for r in exact)}")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in sorted(fail, key=lambda r: -r["overlap"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
