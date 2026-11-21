"""D2 Step 2: Set A diagnostics G1-G4 for the DeBERTa grounding score (CPU, standard library only).

  G1  Is DeBERTa an independent but related checker? Sentence level: agreement / Cohen's kappa with the T5
      verdict and with Self-RAG's own ISSUP label on the graded claims. Candidate level: correlation with exact
      ALCE citation recall.
  G2  Within-question concordance with str-em (pairs of candidates of one question with different str-em; share
      where the higher-scoring candidate is the better one; ties 0.5). 95% bootstrap over questions, 1000 reps,
      seed 6103 (same as contradiction/diagnostics.py). Reference: released score.
  G3  Same with exact citation recall as the target.
      Also "fix rate": among pairs the released score orders WRONG, the share the signal orders right
      (what an added grounding term could repair).
  G4  The 30 candidates where ISSUP and DeBERTa disagree most (first sentence): CSV for hand labelling, with the
      T5 verdict as a reference column.

Grounding aggregations g over a candidate's sentences (p = DeBERTa probabilities, sentence vs own passage):
  min, mean, first: of p_ent over ALCE sentences;  full: p_ent with the whole answer as hypothesis;
  soft_mean: mean of p_ent + 0.5 * p_neu (softens "extra detail -> neutral");  nocon_min: min of 1 - p_con.

    python grounding/diagnostics.py --split critique/splits/asqa_A.json --ground results/runs/cand_ground_A.jsonl \
        --out results/ground_diag_A
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from cand_metrics import norm_claim  # noqa: E402
from honesty_check import support_score  # noqa: E402

SEED = 6103
AGGS = ("min", "mean", "first", "full", "soft_mean", "nocon_min")


def aggregates(sent_rows):
    s = sorted([r for r in sent_rows if r["sent_idx"] >= 0], key=lambda r: r["sent_idx"])
    full = [r for r in sent_rows if r["sent_idx"] == -1]
    ent = [r["p_ent"] for r in s]
    return {"min": min(ent), "mean": sum(ent) / len(ent), "first": ent[0],
            "full": full[0]["p_ent"] if full else ent[0],
            "soft_mean": sum(r["p_ent"] + 0.5 * r["p_neu"] for r in s) / len(s),
            "nocon_min": min(1 - r["p_con"] for r in s)}


def concordance(questions, key, target):
    hits, n = 0.0, 0
    for cands in questions:
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                x, y = cands[i], cands[j]
                if x[target] == y[target]:
                    continue
                hi, lo = (x, y) if x[target] > y[target] else (y, x)
                n += 1
                hits += 1.0 if hi[key] > lo[key] else 0.5 if hi[key] == lo[key] else 0.0
    return (hits / n if n else float("nan")), n


def fix_rate(questions, key, target, base="final_score"):
    """Among pairs with different target that `base` orders wrong (strictly), share that `key` orders right."""
    hits, n = 0.0, 0
    for cands in questions:
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                x, y = cands[i], cands[j]
                if x[target] == y[target]:
                    continue
                hi, lo = (x, y) if x[target] > y[target] else (y, x)
                if hi[base] >= lo[base]:
                    continue
                n += 1
                hits += 1.0 if hi[key] > lo[key] else 0.5 if hi[key] == lo[key] else 0.0
    return (hits / n if n else float("nan")), n


def bootstrap(questions, fn, reps=1000):
    rng = random.Random(SEED)
    vals = []
    for _ in range(reps):
        v, n = fn([questions[rng.randrange(len(questions))] for _ in questions])
        if n:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sx * sy) if sx and sy else float("nan")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="critique/splits/asqa_A.json")
    ap.add_argument("--ground", default="results/runs/cand_ground_A.jsonl")
    ap.add_argument("--metrics", default="results/runs/cand_metrics_AB_exact.jsonl")
    ap.add_argument("--cands", default="results/runs/asqa_cands_0_end.json")
    ap.add_argument("--verdicts", default="results/runs/asqa_cands_0_end.json.verdicts.jsonl")
    ap.add_argument("--out", default="results/ground_diag_A")
    ap.add_argument("--threshold", type=float, default=0.5, help="p_ent threshold for the sentence-level G1 labels")
    a = ap.parse_args()
    assert "asqa_C" not in a.split, "Set C is reserved for the pre-registered test"

    split = set(load_split_indices(a.split))
    ground = collections.defaultdict(list)
    for line in open(a.ground):
        r = json.loads(line)
        ground[(r["q_idx"], r["node_id"])].append(r)
    items = {it["q_idx"]: it for it in json.load(open(a.cands))["data"] if it["q_idx"] in split}
    cand_raw = {(q, c["node_id"]): c for q, it in items.items() for c in it["candidates"] if c["depth"] == 1}

    by_q = collections.defaultdict(list)
    for line in open(a.metrics):
        r = json.loads(line)
        if r["q_idx"] in split and not r["multi_segment"]:
            r.update(aggregates(ground[(r["q_idx"], r["node_id"])]))
            r["cit"] = r["citation_rec"] if r["citation_rec"] is not None else 0.0
            r["t5"] = 0.0 if r["cit_known"] == 0 else r["cit_proxy"] / 100.0
            by_q[r["q_idx"]].append(r)
    qs = [sorted(v, key=lambda r: r["p_idx"]) for _, v in sorted(by_q.items())]
    rep = {"split": a.split, "questions": len(qs), "candidates": sum(len(c) for c in qs)}

    # ---- G1 sentence level: graded claims with a T5 verdict
    verdicts = {}
    for line in open(a.verdicts):
        v = json.loads(line)
        if v["q_idx"] in split and v["depth"] == 1:
            verdicts[(v["q_idx"], v["node_id"], norm_claim(v["claim"]))] = v
    sent = []
    for (q, node), rows in ground.items():
        if q not in split:
            continue
        for r in rows:
            v = verdicts.get((q, node, norm_claim(r["sentence"]))) if r["sent_idx"] >= 0 else None
            if v:
                sent.append({"deb": int(r["p_ent"] >= a.threshold), "t5": v["verdict"],
                             "issup": int(v["label"] == "[Fully supported]"), "p_ent": r["p_ent"]})
    g1 = {"sentences": len(sent), "threshold": a.threshold,
          "rate_supported": {k: round(sum(s[k] for s in sent) / len(sent), 3) for k in ("deb", "t5", "issup")}}
    for x, y in (("deb", "t5"), ("issup", "t5"), ("deb", "issup")):
        g1[f"{x}_vs_{y}"] = {"agree": round(sum(s[x] == s[y] for s in sent) / len(sent), 3),
                             "kappa": round(kappa([s[x] for s in sent], [s[y] for s in sent]), 3)}
    allc = [r for c in qs for r in c]
    g1["candidate_pearson_with_citation_rec"] = {
        k: round(pearson([r[k] for r in allc], [r["cit"] for r in allc]), 3)
        for k in ("ground_score", "final_score", "t5") + AGGS}
    rep["G1"] = g1

    # ---- G2 / G3 concordance (+ bootstrap) and fix rate
    signals = ("final_score", "ground_score", "t5") + AGGS
    for name, target in (("G2_str_em", "str_em"), ("G3_citation_rec", "cit")):
        res = {}
        for k in signals:
            v, n = concordance(qs, k, target)
            fr, nf = fix_rate(qs, k, target)
            res[k] = {"concordance": round(v, 3), "ci95": bootstrap(qs, lambda s, k=k: concordance(s, k, target)),
                      "pairs": n, "fix_rate_vs_released": round(fr, 3) if k != "final_score" else None,
                      "pairs_released_wrong": nf}
        rep[name] = res

    # ---- G4: largest ISSUP vs DeBERTa disagreement on the first sentence
    g4 = []
    for c in allc:
        raw = cand_raw[(c["q_idx"], c["node_id"])]
        if not raw["support_probs"]:
            continue
        issup = support_score(raw["support_probs"][0], 1.0)
        first = next(r for r in ground[(c["q_idx"], c["node_id"])] if r["sent_idx"] == 0)
        v = verdicts.get((c["q_idx"], c["node_id"], norm_claim(first["sentence"])))
        g4.append({"q_idx": c["q_idx"], "p_idx": c["p_idx"], "gap": issup - first["p_ent"], "issup_score": issup,
                   "issup_label": v["label"] if v else max(raw["support_probs"][0], key=raw["support_probs"][0].get),
                   "deberta_p_ent": first["p_ent"], "deberta_p_neu": first["p_neu"], "deberta_p_con": first["p_con"],
                   "t5_verdict": v["verdict"] if v else "", "cand_citation_rec": c["citation_rec"],
                   "question": items[c["q_idx"]]["question"], "sentence": first["sentence"],
                   "passage": "Title: %s\n%s" % (raw["ctx"].get("title", ""), raw["ctx"].get("text", ""))})
    g4.sort(key=lambda r: -abs(r["gap"]))
    top = g4[:30]
    with open(a.out + "_G4.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()) + ["human_supported", "error_type", "note"])
        w.writeheader()
        for r in top:
            w.writerow({**r, "gap": round(r["gap"], 3), "issup_score": round(r["issup_score"], 3)})
    hi_issup = [r for r in top if r["gap"] > 0]
    with_t5 = [r for r in top if r["t5_verdict"] != ""]
    rep["G4"] = {"n": len(top), "issup_high_deberta_low": len(hi_issup), "deberta_high_issup_low": len(top) - len(hi_issup),
                 "t5_sides_with_deberta": sum((r["t5_verdict"] == 1) == (r["deberta_p_ent"] >= 0.5) for r in with_t5),
                 "t5_sides_with_issup": sum((r["t5_verdict"] == 1) == (r["issup_score"] >= 0.5) for r in with_t5),
                 "with_t5": len(with_t5), "csv": a.out + "_G4.csv"}

    json.dump(rep, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
