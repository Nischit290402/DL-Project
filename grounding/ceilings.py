"""D2 Step 0.5 ceilings, CPU only, Set A and Set B (Set C is not looked at).

Input: grounding/cand_metrics.py output for the A+B candidates. Per split:
  1. Headroom: questions whose 5 candidates differ in str-em / in the citation proxy (only those can change).
  2. Oracle: best of the 5 candidates per question (str-em; citation proxy).
  3. T5-verdict re-rank (upper-bound reference, NOT a method: it ranks with the grader's own NLI model):
     support term := T5 verdict of the candidate's graded claim, i.e. score = relevance + verdict (replace) or
     released + mu * verdict (add). Shows how much a "perfect external grounding checker" could move the metrics.
  4. Within-question concordance with str-em (same definition as contradiction/diagnostics.py D2): released score
     and T5 verdict. 0.5 = no signal.

Citation numbers use cit_proxy (exact for fully graded answers, biased upward otherwise); `cit_cov` reports the
share of chosen answers whose every sentence has a verdict.

    python grounding/ceilings.py --metrics results/runs/cand_metrics_AB.jsonl --out results/ground_ceilings_AB.json
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
from common import load_split_indices  # noqa: E402

SEED = 6103
REPS = 2000


def first_verdict(r):
    return 0.0 if r["cit_known"] == 0 else r["cit_proxy"] / 100.0  # mean verdict over graded sentences


def pick(cands, key):
    ranked = sorted(cands, key=key, reverse=True)  # stable: ties keep passage order, as in rerank.py
    chosen = ranked[0]
    if chosen["empty"] and len(ranked) > 1:
        chosen = ranked[1]
    return chosen


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def paired_ci(base, new, rng):
    """95% paired bootstrap interval of mean(new - base) over questions (values may be None -> dropped)."""
    d = [n - b for b, n in zip(base, new) if b is not None and n is not None]
    if not d:
        return None
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(REPS))
    return [round(sum(d) / len(d), 2), round(boots[int(0.025 * REPS)], 2), round(boots[int(0.975 * REPS)], 2)]


def concordance(qs, key, target="str_em"):
    agree, total = 0.0, 0
    for cands in qs:
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a[target] == b[target]:
                    continue
                total += 1
                hi, lo = (a, b) if a[target] > b[target] else (b, a)
                agree += 1.0 if key(hi) > key(lo) else 0.5 if key(hi) == key(lo) else 0.0
    return agree / total if total else None, total


def summarise(qs, chooser, base_rows, rng):
    chosen = [chooser(c) for c in qs]
    out = {"str_em": round(mean(r["str_em"] for r in chosen), 2),
           "rougeLsum": round(mean(r["rougeLsum"] for r in chosen), 2),
           "cit_proxy": round(mean(r["cit_proxy"] for r in chosen), 2),
           "length": round(mean(r["length"] for r in chosen), 1),
           "changed": sum(a["node_id"] != b["node_id"] for a, b in zip(chosen, base_rows))}
    keys = ["str_em", "rougeLsum", "cit_proxy"]
    if EXACT:
        out["citation_rec"] = round(mean(r["citation_rec"] for r in chosen), 2)
        out["citation_prec"] = round(mean(r["citation_prec"] for r in chosen), 2)
        keys += ["citation_rec", "citation_prec"]
    for k in keys:
        out["d_" + k] = paired_ci([r[k] for r in base_rows], [r[k] for r in chosen], rng)
    return out


EXACT = False  # set in main(): True when the metrics file has exact ALCE citation (cand_metrics_*_exact.jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--splits", default="critique/splits/asqa_A.json,critique/splits/asqa_B.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global EXACT
    by_q = collections.defaultdict(list)
    for line in open(a.metrics):
        r = json.loads(line)
        by_q[r["q_idx"]].append(r)
    EXACT = all("citation_rec" in r for rs in by_q.values() for r in rs)
    cit = "citation_rec" if EXACT else "cit_proxy"

    report = {}
    for split in a.splits.split(","):
        rng = random.Random(SEED)
        name = os.path.basename(split).replace(".json", "")
        qs = [sorted(by_q[q], key=lambda r: r["p_idx"]) for q in load_split_indices(split) if q in by_q]
        qs = [c for c in qs if not c[0]["multi_segment"]]
        base = [next(r for r in c if r["selected"]) for c in qs]
        # sanity: the released score must reproduce C1's choice
        assert all(pick(c, lambda r: r["final_score"])["node_id"] == b["node_id"] for c, b in zip(qs, base))

        res = {"n": len(qs)}
        res["headroom"] = {
            "q_str_em_differs": sum(len({r["str_em"] for r in c}) > 1 for c in qs),
            "q_cit_differs": sum(len({r[cit] for r in c if r[cit] is not None}) > 1 for c in qs),
            "q_released_not_best_str_em": sum(b["str_em"] < max(r["str_em"] for r in c) for c, b in zip(qs, base)),
            "q_released_not_best_cit": sum((b[cit] or 0) < max(r[cit] or 0 for r in c) for c, b in zip(qs, base)),
            "mean_candidate_str_em": round(mean(r["str_em"] for c in qs for r in c), 2),
            "mean_candidate_cit": round(mean(r[cit] for c in qs for r in c), 2)}
        res["citation_source"] = cit
        res["released"] = summarise(qs, lambda c: next(r for r in c if r["selected"]), base, rng)
        res["oracle_str_em"] = summarise(qs, lambda c: pick(c, lambda r: (r["str_em"], r["final_score"])), base, rng)
        res["oracle_cit"] = summarise(
            qs, lambda c: pick(c, lambda r: (r[cit] or 0, r["final_score"])), base, rng)
        res["oracle_cit_then_str_em"] = summarise(
            qs, lambda c: pick(c, lambda r: (r[cit] or 0, r["str_em"], r["final_score"])), base, rng)
        res["t5_replace"] = summarise(
            qs, lambda c: pick(c, lambda r: r["relevance_score"] + first_verdict(r)), base, rng)
        for mu in (0.25, 0.5, 1.0):
            res[f"t5_add_mu{mu}"] = summarise(
                qs, lambda c, mu=mu: pick(c, lambda r: r["final_score"] + mu * first_verdict(r)), base, rng)
        c_rel, n_pairs = concordance(qs, lambda r: r["final_score"])
        c_t5, _ = concordance(qs, first_verdict)
        c_sup, _ = concordance(qs, lambda r: r["ground_score"])
        res["concordance_str_em"] = {"pairs": n_pairs, "released_score": round(c_rel, 3),
                                     "issup_ground_score": round(c_sup, 3), "t5_verdict": round(c_t5, 3)}
        if EXACT:
            cq = [[dict(r, _cit=r["citation_rec"] or 0.0) for r in c] for c in qs]
            k_rel, n_cit = concordance(cq, lambda r: r["final_score"], "_cit")
            k_sup, _ = concordance(cq, lambda r: r["ground_score"], "_cit")
            k_t5, _ = concordance(cq, first_verdict, "_cit")
            res["concordance_citation_rec"] = {"pairs": n_cit, "released_score": round(k_rel, 3),
                                               "issup_ground_score": round(k_sup, 3), "t5_verdict": round(k_t5, 3)}
        report[name] = res

    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
