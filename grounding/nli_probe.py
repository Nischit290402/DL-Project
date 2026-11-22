"""Why does DeBERTa miss near-verbatim support? Re-score the failures under controlled variations (GPU, minutes).

Input: results/runs/verbatim_fail_AB.jsonl from grounding/verbatim_check.py (near-verbatim sentences with
p_ent < 0.5 in cand_ground_*.jsonl) plus a control set of near-verbatim sentences that passed.
Variants (one change each vs the production setting of grounding/cand_ground.py):
  prod        slow tokenizer, premise "Title: t\\ntext", fp32                 (must reproduce the stored p_ent)
  fast        fast tokenizer, same premise
  no_title    premise = text only
  window      premise = the passage sentence(s) that contain the copied span, +-1 sentence
  only_match  premise = just the matching passage sentence(s)
  swap        hypothesis first, passage second (wrong order on purpose, sanity check)
Also prints how slow and fast tokenizers differ on these inputs and how many inputs hit 512 tokens.

    python grounding/nli_probe.py --fail results/runs/verbatim_fail_AB.jsonl --out results/runs/nli_probe.json
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verbatim_check import lcs_run, toks  # noqa: E402

MODEL = "cross-encoder/nli-deberta-v3-large"


def windows(text, sent, sent_tokenize):
    ps = sent_tokenize(text)
    st = toks(sent)
    score = [lcs_run(st, toks(p)) for p in ps]
    best = max(range(len(ps)), key=lambda i: score[i])
    # the copied span may straddle two passage sentences: take every sentence that shares a long run
    hit = [i for i in range(len(ps)) if score[i] >= max(3, 0.3 * len(st))] or [best]
    lo, hi = min(hit), max(hit)
    return " ".join(ps[lo:hi + 1]), " ".join(ps[max(0, lo - 1):hi + 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail", required=True)
    ap.add_argument("--control", default=None, help="near-verbatim rows that passed (built from --ground if absent)")
    ap.add_argument("--n_control", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from nltk import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    fail = [json.loads(l) for l in open(a.fail, encoding="utf-8")]
    ctrl = [json.loads(l) for l in open(a.control, encoding="utf-8")] if a.control else []
    random.Random(0).shuffle(ctrl)
    rows = [dict(r, group="fail") for r in fail] + [dict(r, group="pass") for r in ctrl[:a.n_control]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    slow = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    # the cached tokenizer.json is newer than the pinned tokenizers 0.15 can parse; rebuild fast from the spm model
    fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True, from_slow=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    print("labels", model.config.id2label, "| slow", type(slow).__name__, "| fast", type(fast).__name__, flush=True)

    def score(tok, prem, hyp):
        out = []
        with torch.no_grad():
            for s in range(0, len(prem), 16):
                enc = tok(prem[s:s + 16], hyp[s:s + 16], truncation=True, max_length=512, padding=True,
                          return_tensors="pt").to(dev)
                pr = model(**enc).logits.float().softmax(-1).tolist()
                out += [{"ent": p[lab["entailment"]], "neu": p[lab["neutral"]], "con": p[lab["contradiction"]]}
                        for p in pr]
        return out

    full = ["Title: %s\n%s" % (r["title"], r["text"]) for r in rows]
    hyp = [r["sentence"] for r in rows]
    only, win = zip(*[windows(r["text"], r["sentence"], sent_tokenize) for r in rows])
    variants = {
        "prod": score(slow, full, hyp),
        "fast": score(fast, full, hyp),
        "no_title": score(slow, [r["text"] for r in rows], hyp),
        "window": score(slow, list(win), hyp),
        "only_match": score(slow, list(only), hyp),
        "swap": score(slow, hyp, full),
    }
    # tokenizer comparison on the production inputs
    n_diff, n_512, ex = 0, 0, None
    for p, h in zip(full, hyp):
        i_s = slow(p, h, truncation=True, max_length=512)["input_ids"]
        i_f = fast(p, h, truncation=True, max_length=512)["input_ids"]
        n_512 += len(i_s) >= 512
        if i_s != i_f:
            n_diff += 1
            if ex is None:
                k = next(k for k in range(min(len(i_s), len(i_f))) if i_s[k] != i_f[k]) if any(
                    x != y for x, y in zip(i_s, i_f)) else min(len(i_s), len(i_f))
                ex = {"slow": slow.convert_ids_to_tokens(i_s[max(0, k - 5):k + 10]),
                      "fast": fast.convert_ids_to_tokens(i_f[max(0, k - 5):k + 10]), "len": [len(i_s), len(i_f)]}
    lens = [len(slow(p, h)["input_ids"]) for p, h in zip(full, hyp)]

    summary = {"n_fail": len(fail), "n_pass": len(rows) - len(fail), "tokenizer_diff": n_diff,
               "hit_512": n_512, "len_median": sorted(lens)[len(lens) // 2], "len_max": max(lens),
               "tokenizer_example": ex, "variants": {}}
    for name, res in variants.items():
        for g in ("fail", "pass"):
            idx = [i for i, r in enumerate(rows) if r["group"] == g]
            if not idx:
                continue
            ent = [res[i]["ent"] for i in idx]
            summary["variants"].setdefault(name, {})[g] = {
                "mean_ent": round(sum(ent) / len(ent), 3), "share_ent_ge_0.5": round(sum(e >= .5 for e in ent) / len(ent), 3),
                "share_con_top": round(sum(res[i]["con"] > max(res[i]["ent"], res[i]["neu"]) for i in idx) / len(idx), 3)}
    stored = [r["p_ent"] for r in rows]
    summary["prod_reproduces_stored_max_abs_diff"] = round(max(abs(s - v["ent"]) for s, v in zip(stored, variants["prod"])), 4)
    # premise length vs failure in prod
    summary["len_by_group"] = {g: sorted(l for l, r in zip(lens, rows) if r["group"] == g)[
        len([1 for r in rows if r["group"] == g]) // 2] for g in ("fail", "pass") if any(r["group"] == g for r in rows)}
    per_row = [{"q_idx": r["q_idx"], "node_id": r["node_id"], "sent_idx": r["sent_idx"], "group": r["group"],
                "sentence": r["sentence"], "tokens": l, **{k: round(v[i]["ent"], 4) for k, v in variants.items()}}
               for i, (r, l) in enumerate(zip(rows, lens))]
    json.dump({"summary": summary, "rows": per_row}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("PROBE_OK", a.out)


if __name__ == "__main__":
    main()
