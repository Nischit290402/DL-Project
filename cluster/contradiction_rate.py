"""Go/no-go diagnostic for the self-contradiction idea.

How often do Self-RAG's long-form answers contradict themselves *across segments*?
Runs an off-the-shelf NLI model over every pair of segments in each answer (both directions,
p_contra = max of the two) on an existing ASQA output file. No generation, so it is cheap.

Usage on the cluster (never on the login node): submit cluster/run_contra.sbatch, which runs
  python cluster/contradiction_rate.py out/asqa_baseline.json --out_prefix out/contra_baseline
  python cluster/contradiction_rate.py out/asqa_baseline.json --out_prefix out/contra_baseline_q --with_question

Outputs:
  <prefix>_summary.json   answer-level and pair-level contradiction rates at several thresholds
  <prefix>_pairs.csv      every pair with p_contra >= --dump_min, highest first, with an empty
                          `human_label` column: hand-check ~30 of these before trusting any number
                          (ASQA answers legitimately list different answers for different readings
                          of the question, which NLI can mistake for contradiction)
  <prefix>_items.jsonl    per-answer max p_contra, to correlate with str-EM later
"""
import argparse, csv, itertools, json, re

SPECIAL = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]",
           "[Retrieval]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Continue to Use Evidence]",
           "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "</s>"]
THRESHOLDS = [0.5, 0.7, 0.9]


def clean_segment(raw):
    """Mirror run_long_form_static.py: cut at the next [Retrieval], then strip reflection tokens."""
    if "[Retrieval]" in raw:
        raw = raw[:raw.index("[Retrieval]")]
    raw = re.sub(r"<paragraph>.*?</paragraph>", "", raw, flags=re.S)
    for tok in SPECIAL:
        raw = raw.replace(tok, "")
    return " ".join(raw.split())


def get_segments(item):
    """Segments of the selected beam path; falls back to splitting the cited output."""
    if item.get("intermediate"):
        segs = [clean_segment(s) for s in item["intermediate"]]
    else:
        segs = [s.strip() for s in re.split(r"\s*\[\d+\]\s*\.?\s*", item.get("output", ""))]
    out, seen = [], set()
    for s in segs:
        if len(s) >= 3 and s not in seen:  # same dedup as the inference script
            seen.add(s)
            out.append(s)
    return out


def load_nli(model_name, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # tokenizer.json needs tokenizers>=0.19; env pins 0.15
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    c_idx = labels["contradiction"]

    @torch.no_grad()
    def score(premises, hypotheses, batch_size=32):
        probs = []
        for i in range(0, len(premises), batch_size):
            enc = tok(premises[i:i + batch_size], hypotheses[i:i + batch_size], truncation=True,
                      max_length=512, padding=True, return_tensors="pt").to(device)
            p = model(**enc).logits.float().softmax(-1)[:, c_idx]
            probs.extend(p.tolist())
        return probs
    return score


def analyse(data, score_fn, with_question=False, dump_min=0.5):
    pairs, items = [], []
    prem, hyp, meta = [], [], []
    for idx, item in enumerate(data):
        segs = get_segments(item)
        ctx = ("Question: " + item["question"] + " Answer: ") if with_question else ""
        items.append({"idx": idx, "question": item["question"], "n_segments": len(segs), "max_p_contra": 0.0})
        for i, j in itertools.combinations(range(len(segs)), 2):
            prem += [ctx + segs[i], ctx + segs[j]]
            hyp += [segs[j], segs[i]]
            meta.append((idx, i, j, segs[i], segs[j]))
    probs = score_fn(prem, hyp) if prem else []
    for k, (idx, i, j, si, sj) in enumerate(meta):
        p = max(probs[2 * k], probs[2 * k + 1])
        items[idx]["max_p_contra"] = max(items[idx]["max_p_contra"], p)
        pairs.append({"idx": idx, "question": data[idx]["question"], "seg_i": i, "seg_j": j,
                      "text_i": si, "text_j": sj, "p_contra": round(p, 4)})

    multi = [it for it in items if it["n_segments"] >= 2]
    summary = {
        "n_answers": len(items),
        "n_answers_with_2plus_segments": len(multi),
        "mean_segments": round(sum(it["n_segments"] for it in items) / max(len(items), 1), 2),
        "n_pairs": len(pairs),
        "with_question_context": with_question,
        "answer_rate": {str(t): round(sum(it["max_p_contra"] >= t for it in multi) / max(len(multi), 1), 4)
                        for t in THRESHOLDS},
        "pair_rate": {str(t): round(sum(p["p_contra"] >= t for p in pairs) / max(len(pairs), 1), 4)
                      for t in THRESHOLDS},
        "note": "answer_rate = share of multi-segment answers with at least one pair above the threshold",
    }
    flagged = sorted([p for p in pairs if p["p_contra"] >= dump_min], key=lambda p: -p["p_contra"])
    return summary, flagged, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ASQA output json from run_long_form_static.py (or merged shards)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-large")
    ap.add_argument("--with_question", action="store_true",
                    help="prefix the question to the premise, so different readings are less likely to look contradictory")
    ap.add_argument("--dump_min", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.load(open(a.input))["data"][:a.limit]
    summary, flagged, items = analyse(data, load_nli(a.model, device), a.with_question, a.dump_min)
    summary.update({"input": a.input, "model": a.model})

    json.dump(summary, open(a.out_prefix + "_summary.json", "w"), indent=2)
    with open(a.out_prefix + "_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "question", "seg_i", "seg_j", "text_i", "text_j", "p_contra", "human_label"])
        w.writeheader()
        for p in flagged:
            w.writerow({**p, "human_label": ""})
    with open(a.out_prefix + "_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(json.dumps(summary, indent=2))
    print("flagged pairs (p >= {}): {} -> {}_pairs.csv".format(a.dump_min, len(flagged), a.out_prefix))


if __name__ == "__main__":
    main()
