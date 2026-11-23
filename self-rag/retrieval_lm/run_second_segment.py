"""Second segment (GPU). Self-RAG's long-form design builds an answer from several segments, each written from
its own passage. On ASQA the released checkpoint ends 947 of 948 answers after one segment. This script makes the
model take the step it was built for: given the first segment an answer already has, it writes one continuation
per remaining passage, exactly as run_long_form_static.py would at depth 2 (run_step_generation_batch, same
prompt convention: root prompt + previous generation + [Retrieval]<paragraph>...</paragraph>).

Nothing is chosen here. Every continuation is saved with its reflection-token scores in the same schema as
--save_candidates (depth 1 entries), so the divisions' scoring scripts run on the output unchanged.

  --picks   answer file whose items carry q_idx and chosen_p_idx (the first segment), e.g. the combined system
  --cands   the --save_candidates files those picks came from (comma-separated)
  --with_first_paragraph  also keep the first segment's passage in the prompt (the format used in training);
                          default is the released script's convention, which drops it
"""
import argparse, json, re, time
import run_long_form_static as R
from transformers import AutoTokenizer
from vllm import LLM
from utils import TASK_INST, PROMPT_DICT, load_special_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name"); ap.add_argument("--download_dir", default=".cache")
    ap.add_argument("--picks", required=True); ap.add_argument("--cands", required=True)
    ap.add_argument("--indices_file", required=True); ap.add_argument("--output_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=300); ap.add_argument("--dtype", default="half")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--with_first_paragraph", action="store_true")
    ap.add_argument("--max_questions", type=int, default=None)
    a = ap.parse_args()
    R.KEEP_SPECIAL_TOKENS = True; R.SAVE_CANDIDATES = True
    tok = AutoTokenizer.from_pretrained(a.model_name, padding_side="left")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(tok, use_grounding=True, use_utility=True)
    model = LLM(model=a.model_name, download_dir=a.download_dir, dtype=a.dtype, tensor_parallel_size=1)
    keep = set()
    for p in a.indices_file.split(","): keep |= set(json.load(open(p))["indices"])
    cands = {}
    for f in a.cands.split(","):
        for it in json.load(open(f))["data"]: cands[int(it["q_idx"])] = it
    picks = [x for x in json.load(open(a.picks))["data"] if int(x["q_idx"]) in keep and "chosen_p_idx" in x]
    picks = sorted(picks, key=lambda x: int(x["q_idx"]))[:a.max_questions]
    print("questions:", len(picks), "with_first_paragraph =", a.with_first_paragraph, flush=True)
    out, t0 = [], time.time()
    for n, pk in enumerate(picks):
        q = int(pk["q_idx"]); item = cands[q]
        d1 = sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])
        first = next(c for c in d1 if c["p_idx"] == int(pk["chosen_p_idx"])); others = [c for c in d1 if c is not first]
        prev = first["pred"].replace("</s>", "")
        prev = re.sub(r"\[Utility:[1-5]\]\s*$", "", prev)          # the utility token closes an answer; drop it to continue
        root = PROMPT_DICT["prompt_no_input"].format_map({"instruction": TASK_INST["asqa"] + "## Input:\n\n" + item["question"]})
        if a.with_first_paragraph:
            root += "[Retrieval]<paragraph>{}</paragraph>".format(first["ctx"]["title"] + "\n" + first["ctx"]["text"])
        preds, scores, osd = R.run_step_generation_batch(
            model, root + prev, [c["ctx"] for c in others], a.max_new_tokens, rel_tokens, grd_tokens=grd_tokens,
            ret_tokens=ret_tokens, ut_tokens=ut_tokens, threshold=a.threshold, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=True)
        new = {k: item[k] for k in ("question", "qa_pairs", "annotations", "answer", "sample_id") if k in item}
        new.update(q_idx=q, first_p_idx=first["p_idx"], first_node_id=first["node_id"], first_pred=first["pred"], candidates=[])
        for i, (c, pred) in enumerate(zip(others, preds)):
            proc = pred[:pred.index("[Retrieval]")] if "[Retrieval]" in pred else pred
            new["candidates"].append({"node_id": c["node_id"], "parent": first["node_id"], "depth": 1, "p_idx": c["p_idx"],
                                      "pred": pred, "processed_pred": proc, "beam_score": float(scores[i]),
                                      "ctx": {"title": c["ctx"].get("title", ""), "text": c["ctx"].get("text", "")}, **osd[i]["saved"]})
        out.append(new)
        if n % 20 == 0:
            el = time.time() - t0; print(f"[{n + 1}/{len(picks)}] {el / 60:.1f} min, {el / (n + 1):.1f} s/question", flush=True)
            json.dump({"data": out}, open(a.output_file + "_tmp", "w"))
    json.dump({"data": out, "args": vars(a)}, open(a.output_file, "w"))
    print("SECOND_SEGMENT_OK", a.output_file, len(out), flush=True)


if __name__ == "__main__":
    main()
