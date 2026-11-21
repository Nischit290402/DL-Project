"""Per-candidate ASQA metrics, CPU only (D2 Step 0.5).

For every depth-1 candidate of every question, build the exact answer text Self-RAG would output if that candidate
were chosen (critique/rerank.py build_output) and score it like critique/score_per_question.py:

  str_em, length, rougeLsum   exact ALCE computation (re-implemented, no ALCE checkout or GPU needed)
  cit_sents                   number of ALCE sentences (nltk sent_tokenize) in the answer
  cit_known / cit_proxy       citation-recall proxy from the saved T5 verdicts (critique/nli_verdicts.py):
                              a sentence's verdict is known when its text equals a claim Self-RAG graded with
                              [IsSup]; cit_proxy = 100 * mean verdict over known sentences (None if none known).
                              Only the first sentence is usually graded, so this is a partial view of ALCE's
                              citation recall (ALCE checks every sentence with T5-11B). Validated in ceilings.py.

Writes one JSON line per candidate: q_idx, p_idx, node_id, selected (= C1's choice), the metrics above.

    python grounding/cand_metrics.py --f results/runs/asqa_cands_0_end.json \
        --verdicts results/runs/asqa_cands_0_end.json.verdicts.jsonl --out results/runs/cand_metrics_AB.jsonl
    python grounding/cand_metrics.py --f results/runs/asqa_cands_C.json --out results/runs/cand_metrics_C.jsonl
"""
import argparse
import collections
import json
import os
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
from common import postprocess  # noqa: E402
from rerank import build_output  # noqa: E402


def normalize_answer(s):
    """ALCE utils.normalize_answer."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def remove_citations(sent):
    """ALCE utils.remove_citations."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def exact_presence(short_answers, context):
    """ALCE eval.exact_presence."""
    ctx = normalize_answer(context)
    return any(normalize_answer(a) in ctx for a in short_answers)


def norm_claim(s):
    return " ".join(remove_citations(s).split()).strip().rstrip(".").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="--save_candidates output file")
    ap.add_argument("--verdicts", default="", help="T5 verdicts for the candidates (nli_verdicts.py output)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from nltk import sent_tokenize
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    verdicts = collections.defaultdict(dict)  # (q_idx, node_id) -> {normalised claim: verdict}
    if a.verdicts:
        for line in open(a.verdicts):
            r = json.loads(line)
            if r["depth"] == 1:
                verdicts[(r["q_idx"], r["node_id"])][norm_claim(r["claim"])] = r["verdict"]

    data = json.load(open(a.f))["data"]
    n = 0
    with open(a.out, "w") as f:
        for item in data:
            path = item["best_paths"][str(item.get("output_path", 0))]
            n_segments = len([x for x in path if x not in (None, 0)])
            refs = [x["long_answer"] for x in item["annotations"][:2]] if item.get("annotations") else [item["answer"]]
            refs = ["\n".join(sent_tokenize(r.lower())) for r in refs]
            for c in sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"]):
                out = build_output(item, c).strip().split("\n")[0].replace("<|im_end|>", "")  # clean_like_alce
                no_cite = remove_citations(out)
                hits = [exact_presence(p["short_answers"], no_cite) for p in item["qa_pairs"]]
                hyp = "\n".join(sent_tokenize(no_cite.lower()))
                sents = sent_tokenize(out)
                known = [verdicts[(item["q_idx"], c["node_id"])].get(norm_claim(s)) for s in sents]
                known = [v for v in known if v is not None]
                row = {"q_idx": item["q_idx"], "p_idx": c["p_idx"], "node_id": c["node_id"],
                       "selected": c["node_id"] == path[-1], "multi_segment": n_segments > 1,
                       "empty": len(postprocess(" " + c["processed_pred"])) == 0,
                       "final_score": c["final_score"], "ground_score": c["ground_score"],
                       "relevance_score": c["relevance_score"],
                       "str_em": 100.0 * sum(hits) / len(hits), "length": len(no_cite.split()),
                       "rougeLsum": 100.0 * max(scorer.score(r, hyp)["rougeLsum"].fmeasure for r in refs),
                       "cit_sents": len(sents), "cit_known": len(known),
                       "cit_proxy": 100.0 * sum(known) / len(known) if known else None}
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"{n} candidates from {len(data)} questions -> {a.out}")


if __name__ == "__main__":
    main()
