"""Self-contradiction across candidates: go/no-go diagnostics D1-D3 (laptop, Python standard library only).

  D1  How common: share of questions with at least one contradicting candidate pair (p_con >= 0.5 / 0.7 / 0.9).
  D2  Does it predict errors: within each question, is a candidate that the others contradict more (higher C)
      less correct (lower str-em)? Reported as within-question concordance with a 95% bootstrap interval:
      0.5 = no signal, 1.0 = lower C always means higher str-em. Compared with the same measure for Self-RAG's
      own released score and for a word-overlap agreement score.
  D3  False alarms: writes every flagged pair to a CSV with a random sample of 30 marked for hand-labelling.

C_i = mean over the other candidates j of max(p_con(i->j), p_con(j->i)).
str-em per candidate = ALCE's str-em on that candidate's text (same normalisation as ALCE eval.py).

    python3 contradiction/diagnostics.py --nli results/runs/cand_nli_A.jsonl \
        --f results/runs/asqa_cands_0_end.json --split critique/splits/asqa_A.json --out results/cand_contra_A
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "critique"))
from common import clean_claim, load_split_indices  # noqa: E402

THRESHOLDS = (0.5, 0.7, 0.9)
SEED = 6103


def normalize_answer(s):
    """ALCE eval.py normalize_answer."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def str_em(text, qa_pairs):
    t = normalize_answer(text)
    return sum(any(normalize_answer(a) in t for a in qa["short_answers"]) for qa in qa_pairs) / len(qa_pairs)


def token_f1(a, b):
    a, b = a.split(), b.split()
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def concordance(questions, key):
    """Within-question pairs with different str-em: share where the candidate with lower `key` has higher str-em.
    Ties in key count 0.5. Returns (value, number of pairs)."""
    hits, n = 0.0, 0
    for cands in questions:
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                x, y = cands[a], cands[b]
                if x["em"] == y["em"]:
                    continue
                hi, lo = (x, y) if x["em"] > y["em"] else (y, x)
                n += 1
                hits += 1.0 if hi[key] < lo[key] else 0.5 if hi[key] == lo[key] else 0.0
    return (hits / n if n else float("nan")), n


def bootstrap(questions, key, reps=1000):
    rng = random.Random(SEED)
    vals = []
    for _ in range(reps):
        sample = [questions[rng.randrange(len(questions))] for _ in questions]
        v, n = concordance(sample, key)
        if n:
            vals.append(v)
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", required=True)
    ap.add_argument("--f", required=True)
    ap.add_argument("--split", default="critique/splits/asqa_A.json")
    ap.add_argument("--out", required=True, help="prefix: writes <out>.json and <out>_pairs.csv")
    a = ap.parse_args()

    keep = set(load_split_indices(a.split))
    items = {it["q_idx"]: it for it in json.load(open(a.f))["data"] if it["q_idx"] in keep}
    nli = collections.defaultdict(dict)  # (q, variant) -> {(i, j): row}
    for line in open(a.nli):
        r = json.loads(line)
        nli[(r["q_idx"], r["variant"])][(r["i"], r["j"])] = r

    summary = {"split": a.split, "n_questions": len(items), "nli_file": a.nli}
    per_variant = {}
    for variant in ("noctx", "ctx"):
        questions, pairs = [], []
        for q, it in sorted(items.items()):
            node = it["best_paths"][str(it.get("output_path", 0))][-1]
            cands = sorted([c for c in it["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])
            qwords = set(normalize_answer(it["question"]).split())
            rows = []
            for c in cands:
                text = clean_claim(c["processed_pred"])
                rows.append({"q_idx": q, "p_idx": c["p_idx"], "text": text, "em": str_em(text, it["qa_pairs"]),
                             "score": c["final_score"], "chosen": c["node_id"] == node,
                             "words": " ".join(w for w in normalize_answer(text).split() if w not in qwords)})
            table = nli[(q, variant)]
            for x in rows:
                cons, ents = [], []
                for y in rows:
                    if y is x:
                        continue
                    cons.append(max(table[(x["p_idx"], y["p_idx"])]["p_con"], table[(y["p_idx"], x["p_idx"])]["p_con"]))
                    ents.append(max(table[(x["p_idx"], y["p_idx"])]["p_ent"], table[(y["p_idx"], x["p_idx"])]["p_ent"]))
                x["C"] = sum(cons) / len(cons)
                x["E"] = sum(ents) / len(ents)
                x["neg_score"] = -x["score"]
                x["neg_overlap"] = -sum(token_f1(x["words"], y["words"]) for y in rows if y is not x) / (len(rows) - 1)
            for ia in range(len(rows)):
                for ib in range(ia + 1, len(rows)):
                    x, y = rows[ia], rows[ib]
                    p = max(table[(x["p_idx"], y["p_idx"])]["p_con"], table[(y["p_idx"], x["p_idx"])]["p_con"])
                    pairs.append({"q_idx": q, "x": x, "y": y, "p": p,
                                  "short_answers": " | ".join("; ".join(qa["short_answers"]) for qa in it["qa_pairs"])})
            questions.append(rows)

        # D1
        d1 = {}
        for t in THRESHOLDS:
            d1[str(t)] = {
                "questions_with_a_contradicting_pair": round(sum(any(p["p"] >= t for p in pairs if p["q_idx"] == qs[0]["q_idx"])
                                                                for qs in questions) / len(questions), 3),
                "pair_rate": round(sum(p["p"] >= t for p in pairs) / len(pairs), 3),
                "chosen_answer_in_a_contradicting_pair": round(sum(any(p["p"] >= t and (p["x"]["chosen"] or p["y"]["chosen"])
                                                                      for p in pairs if p["q_idx"] == qs[0]["q_idx"])
                                                                  for qs in questions) / len(questions), 3)}
        # D2
        d2 = {}
        for key, name in (("C", "contradiction C (lower = better)"), ("neg_score", "Self-RAG released score"),
                          ("neg_overlap", "word-overlap agreement"), ("E", "entailment E (inverted below)")):
            if key == "E":
                for qs in questions:
                    for x in qs:
                        x["neg_E"] = -x["E"]
                key = "neg_E"
                name = "entailment agreement E (higher = better)"
            v, n = concordance(questions, key)
            d2[name] = {"concordance": round(v, 3), "ci95": bootstrap(questions, key), "pairs": n}
        buckets = collections.OrderedDict((b, []) for b in ("0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-1.0"))
        for qs in questions:
            for x in qs:
                b = "0.0-0.1" if x["C"] < 0.1 else "0.1-0.3" if x["C"] < 0.3 else "0.3-0.5" if x["C"] < 0.5 else "0.5-1.0"
                buckets[b].append(x["em"])
        d2["str_em_by_C_bucket"] = {b: {"n": len(v), "mean_str_em": round(100 * sum(v) / len(v), 2) if v else None}
                                    for b, v in buckets.items()}
        outlier = [qs for qs in questions if max(qs, key=lambda x: x["C"])["chosen"]]
        d2["chosen_is_most_contradicted"] = {
            "questions": len(outlier),
            "chosen_str_em": round(100 * sum(next(x for x in qs if x["chosen"])["em"] for qs in outlier) / max(len(outlier), 1), 2),
            "least_contradicted_str_em": round(100 * sum(min(qs, key=lambda x: x["C"])["em"] for qs in outlier) / max(len(outlier), 1), 2)}
        all_em = [x["em"] for qs in questions for x in qs]
        d2["reference"] = {"chosen_str_em": round(100 * sum(next(x for x in qs if x["chosen"])["em"] for qs in questions) / len(questions), 2),
                           "best_of_5_str_em": round(100 * sum(max(x["em"] for x in qs) for qs in questions) / len(questions), 2),
                           "mean_candidate_str_em": round(100 * sum(all_em) / len(all_em), 2)}
        per_variant[variant] = {"D1": d1, "D2": d2}

        # D3: flagged pairs, random 30 marked for labelling
        if variant == "ctx":
            flagged = sorted([p for p in pairs if p["p"] >= 0.5], key=lambda p: -p["p"])
            sample = set(random.Random(SEED).sample(range(len(flagged)), min(30, len(flagged))))
            noctx = {(p["q_idx"], p["x"]["p_idx"], p["y"]["p_idx"]): p["p"] for p in per_variant.get("_noctx_pairs", [])}
            with open(a.out + "_pairs.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["sample30", "q_idx", "question", "short_answers", "cand_a", "cand_b", "str_em_a", "str_em_b",
                            "p_con_ctx", "p_con_noctx", "label", "note"])
                for k, p in enumerate(flagged):
                    w.writerow([int(k in sample), p["q_idx"], items[p["q_idx"]]["question"], p["short_answers"],
                                p["x"]["text"], p["y"]["text"], round(p["x"]["em"], 3), round(p["y"]["em"], 3), round(p["p"], 3),
                                round(noctx.get((p["q_idx"], p["x"]["p_idx"], p["y"]["p_idx"]), float("nan")), 3), "", ""])
            summary["D3"] = {"flagged_pairs_ctx_p>=0.5": len(flagged), "marked_for_labelling": len(sample),
                             "csv": a.out + "_pairs.csv",
                             "labels": "genuine = the two answers cannot both be true for the same reading of the question; "
                                       "different-reading = both can be true, they answer different readings; "
                                       "nli-error = no real conflict (paraphrase, extra detail, unrelated text)"}
        else:
            per_variant["_noctx_pairs"] = pairs

    per_variant.pop("_noctx_pairs", None)
    summary.update(per_variant)
    json.dump(summary, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
