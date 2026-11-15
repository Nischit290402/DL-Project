"""Score an ASQA output file question by question with ALCE's own metric code.

ALCE's eval.py prints only averages, and a 95% bootstrap interval needs one score per question. This script
imports eval.py from the ALCE checkout (same commit as cluster/setup_alce.sh), applies the same input cleaning as
its main(), and calls ALCE's metric functions one question at a time:

  str_em, length, rougeLsum   CPU
  citation_rec, citation_prec GPU (T5-11B NLI, google/t5_xxl_true_nli_mixture), skip with --no_citations
  mauve                        GPU (gpt2-large), whole-file metric: one number, no per-question value (--mauve)

Averages of the per-question values equal ALCE's numbers for str-em and citations. ROUGE-Lsum differs slightly
because ALCE reports the median of a 1000-sample bootstrap; the per-question mean is what we bootstrap.

Writes <out>.perq.jsonl (one line per question) and <out>.summary.json (averages).
GPU job: sbatch --export=ALL,F=out/asqa_keeptok_0_end.json cluster/run_critique_score.sbatch
"""
import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_outputs, load_split_indices  # noqa: E402


def load_alce(alce_dir):
    sys.path.insert(0, alce_dir)
    import eval as alce_eval  # ALCE's eval.py
    import utils as alce_utils  # ALCE's utils.py (remove_citations, normalize_answer)
    import logging
    alce_eval.logger.setLevel(logging.WARNING)  # ALCE logs every sentence at INFO level
    return alce_eval, alce_utils


def clean_like_alce(item):
    """Same cleaning as ALCE eval.py main(): first line only, no '<|im_end|>'."""
    item = copy.deepcopy(item)
    item["output"] = item["output"].strip().split("\n")[0].replace("<|im_end|>", "")
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", required=True, help="ALCE-format output file ({'data': [...]})")
    parser.add_argument("--split", default="", help="comma-separated split files; default: all questions")
    parser.add_argument("--out", default=None, help="output prefix (default: the input path, plus the split names)")
    parser.add_argument("--alce_dir", default=os.environ.get("ALCE_DIR", os.path.expanduser("~/ALCE")))
    parser.add_argument("--no_citations", action="store_true", help="skip the T5-11B citation metrics")
    parser.add_argument("--mauve", action="store_true", help="also compute MAUVE for the whole file (one number)")
    args = parser.parse_args()

    alce_eval, alce_utils = load_alce(args.alce_dir)
    from rouge_score import rouge_scorer
    from nltk import sent_tokenize

    outputs = load_outputs(args.f)
    if args.split:
        wanted = load_split_indices(args.split)
        missing = [i for i in wanted if i not in outputs]
        assert not missing, f"{len(missing)} split questions are not in {args.f}, e.g. {missing[:5]}"
        q_ids = wanted
    else:
        q_ids = sorted(outputs)
    items = [clean_like_alce(outputs[i]) for i in q_ids]

    if args.out is None:
        tag = "_".join(os.path.basename(p).replace(".json", "") for p in args.split.split(",") if p)
        args.out = args.f + (f".{tag}" if tag else "")

    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    rows = []
    for q, item in zip(q_ids, items):
        no_cite = alce_utils.remove_citations(item["output"])
        row = {"q_idx": q, "length": len(no_cite.split())}
        # str-em, as in ALCE compute_str_em, for one question
        hits = [alce_eval.exact_presence(p["short_answers"], no_cite) for p in item["qa_pairs"]]
        row["str_em"] = 100.0 * sum(hits) / len(hits)
        # ROUGE-Lsum against both ASQA long answers, best of the two, as in ALCE compute_rouge
        hyp = "\n".join(sent_tokenize(no_cite.lower()))
        refs = [a["long_answer"] for a in item["annotations"][:2]] if item.get("annotations") else [item["answer"]]
        row["rougeLsum"] = 100.0 * max(scorer.score("\n".join(sent_tokenize(r.lower())), hyp)["rougeLsum"].fmeasure
                                       for r in refs)
        rows.append(row)

    if not args.no_citations:
        for row, item in zip(rows, items):
            # ALCE skips answers with no sentences; their value stays None and is left out of the average
            res = alce_eval.compute_autoais([item], at_most_citations=3)
            for key in ("citation_rec", "citation_prec"):
                row[key] = None if math.isnan(res[key]) else float(res[key])

    summary = {"file": args.f, "split": args.split or "all", "n": len(rows)}
    for key in ("str_em", "rougeLsum", "length", "citation_rec", "citation_prec"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            summary[key] = sum(vals) / len(vals)
    if args.mauve:
        normalized = [dict(it, output=alce_utils.remove_citations(it["output"])) for it in items]
        summary["mauve"] = alce_eval.compute_mauve(normalized)

    with open(args.out + ".perq.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("PERQ_FILE", args.out + ".perq.jsonl")


if __name__ == "__main__":
    main()
