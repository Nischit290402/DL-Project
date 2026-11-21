"""Changed-question breakdown for a D2 re-ranked file vs the released choice (pre-registered "also reported").

Counts changed picks (different chosen node) and changed answer texts, and on the questions whose answer text
changed compares released vs new citation recall, str-em, ROUGE-Lsum, fully supported (citation recall 100),
sentence count, and per-question better/worse counts. Same table layout as results/nli_grounding_setB.md.

python grounding/changed_breakdown.py --base out/rr_released_lookup_C.json --new out/rr_ground_C.json
"""
import argparse
import json

import nltk

K = ("citation_rec", "str_em", "rougeLsum")


def load(path):
    data = {str(d["q_idx"]): d for d in json.load(open(path, encoding="utf-8"))["data"]}
    perq = {str(r["q_idx"]): r for r in map(json.loads, open(path + ".perq.jsonl", encoding="utf-8"))}
    return data, perq


def n_sents(text):
    return len(nltk.sent_tokenize(text)) if text.strip() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    bd, bp = load(a.base)
    nd, np_ = load(a.new)
    assert nd.keys() == np_.keys() and set(nd) <= set(bd) and set(nd) <= set(bp), "question sets differ"
    bd = {q: bd[q] for q in nd}
    picks = [q for q in bd if str(bd[q].get("chosen_node_id")) != str(nd[q].get("chosen_node_id"))]
    texts = [q for q in bd if bd[q]["output"].strip() != nd[q]["output"].strip()]
    n = len(texts)

    def mean(src, k):
        return sum(src[q][k] for q in texts) / n

    res = {"questions": len(bd), "changed_picks": len(picks), "changed_texts": n, "on_changed_texts": {}}
    for k in K:
        res["on_changed_texts"][k] = {
            "released": mean(bp, k), "new": mean(np_, k),
            "better": sum(np_[q][k] > bp[q][k] + 1e-9 for q in texts),
            "worse": sum(np_[q][k] < bp[q][k] - 1e-9 for q in texts)}
    res["on_changed_texts"]["fully_supported"] = {
        "released": sum(bp[q]["citation_rec"] >= 100 - 1e-9 for q in texts),
        "new": sum(np_[q]["citation_rec"] >= 100 - 1e-9 for q in texts)}
    res["on_changed_texts"]["sentences"] = {
        "released": sum(n_sents(bd[q]["output"]) for q in texts) / n,
        "new": sum(n_sents(nd[q]["output"]) for q in texts) / n}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
