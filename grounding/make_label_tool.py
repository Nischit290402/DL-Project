"""Build the hand-labelling page for (1) D2 answer swaps on Set A+B and (2) the G4 sheet (CPU, seconds).

Task R (ROUGE drops): every A+B question where D2 picked a different answer text than the released run, sorted by
ROUGE-Lsum change (released -> D2). The 30 largest drops are marked as the priority set; the rest are optional.
Task G (G4): the 30 rows of results/ground_diag_A_G4.csv (ISSUP vs DeBERTa disagreement on the first sentence).

Writes one self-contained page, results/label_tool/index.html (template: grounding/label_tool_template.html). Labels autosave in the
browser (localStorage) and are exported with the page's "Export" button to CSV/JSON.

    python grounding/make_label_tool.py
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from diagnostics import aggregates  # noqa: E402

TAG = re.compile(r"\[(Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|"
                 r"Continue to Use Evidence|No Retrieval|Retrieval|Utility:\d)\]|</s>")
N_PRIORITY = 30


def p(*a):
    return os.path.join(ROOT, *a)


def clean(pred):
    return re.sub(r"\s+", " ", TAG.sub(" ", pred)).strip()


def model_labels(out_dir):
    """Pre-filled labels (results/label_tool/claude_labels.json) in the page's format; empty if the file is absent."""
    path = os.path.join(out_dir, "claude_labels.json")
    if not os.path.exists(path):
        return {}
    j = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, (reason, better, tags, note) in j.get("R", {}).items():
        out[k] = {"reason": reason, "better": better, "tags": tags, "note": note, "by": j["by"]}
    for k, (sup, err, note) in j.get("G", {}).items():
        out[k] = {"supported": sup, "error": err, "note": note, "by": j["by"]}
    return out


def write_csvs(out_dir, swaps, g4, lab):
    with open(os.path.join(out_dir, "labels_rouge_swaps_AB.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "set", "q_idx", "rank", "priority", "question", "d_rouge", "d_strem", "d_cit",
                    "released_answer", "d2_answer", "reason", "better", "tags", "note", "by"])
        for x in swaps:
            l = lab.get(x["id"], {})
            w.writerow([x["id"], x["set"], x["q_idx"], x["rank"], int(x["priority"]), x["question"],
                        round(x["d_rouge"], 2), round(x["d_strem"], 2), round(x["d_cit"], 2), x["released"]["text"],
                        x["d2"]["text"], l.get("reason", ""), l.get("better", ""), ";".join(l.get("tags", [])),
                        l.get("note", ""), l.get("by", "")])
    with open(os.path.join(out_dir, "labels_G4.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_idx", "p_idx", "gap", "issup_score", "deberta_p_ent", "deberta_p_con", "t5_verdict",
                    "question", "sentence", "human_supported", "error_type", "note", "by"])
        for x in g4:
            l = lab.get(x["id"], {})
            w.writerow([x["q_idx"], x["p_idx"], x["gap"], x["issup"], x["p_ent"], x["p_con"], x["t5"], x["question"],
                        x["sentence"], l.get("supported", ""), l.get("error", ""), l.get("note", ""), l.get("by", "")])


def main():
    cands = {it["q_idx"]: it for it in json.load(open(p("results/runs/asqa_cands_0_end.json"), encoding="utf-8"))["data"]}
    metrics = {}
    for line in open(p("results/runs/cand_metrics_AB_exact.jsonl")):
        r = json.loads(line)
        metrics[(r["q_idx"], int(r["node_id"]))] = r
    ground_rows = {}
    for s in "AB":
        for line in open(p(f"results/runs/cand_ground_{s}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            ground_rows.setdefault((r["q_idx"], int(r["node_id"])), []).append(r)

    def cand_view(q, node):
        it = cands[q]
        c = next(c for c in it["candidates"] if int(c["node_id"]) == node)
        m = metrics.get((q, node), {})
        g = ground_rows.get((q, node), [])
        return {
            "node_id": node, "p_idx": int(c["p_idx"]), "text": clean(c["processed_pred"]),
            "passage_title": c["ctx"]["title"], "passage": c["ctx"]["text"],
            "rouge": m.get("rougeLsum"), "str_em": m.get("str_em"), "cit_rec": m.get("citation_rec"),
            "cit_prec": m.get("citation_prec"), "length": m.get("length"),
            "issup": float(c["ground_score"]), "relevance": float(c["relevance_score"]),
            "released_score": float(c["final_score"]),
            "g_mean": aggregates(g)["mean"] if g else None,
            "sents": [{"s": r["sentence"], "p_ent": r["p_ent"], "p_con": r["p_con"]}
                      for r in sorted(g, key=lambda r: r["sent_idx"])],
        }

    swaps = []
    for s in "AB":
        for it in json.load(open(p(f"out/rr_ground_{s}.json"), encoding="utf-8"))["data"]:
            q = it["q_idx"]
            if "chosen_node_id" not in it:
                continue  # multi-segment, kept as released
            new = int(it["chosen_node_id"])
            old = next((n for (qq, n), m in metrics.items() if qq == q and m.get("selected")), None)
            if old is None or old == new:
                continue
            rel, d2 = cand_view(q, old), cand_view(q, new)
            if rel["text"] == d2["text"]:
                continue
            ci = cands[q]
            swaps.append({
                "id": f"R-{q}", "task": "R", "set": s, "q_idx": q, "question": ci["question"],
                "short_answers": [{"q": qa["question"], "a": qa["short_answers"]} for qa in ci["qa_pairs"]],
                "references": [a["long_answer"] for a in ci["annotations"] if a.get("long_answer")],
                "released": rel, "d2": d2,
                "d_rouge": (d2["rouge"] or 0) - (rel["rouge"] or 0),
                "d_strem": (d2["str_em"] or 0) - (rel["str_em"] or 0),
                "d_cit": (d2["cit_rec"] or 0) - (rel["cit_rec"] or 0),
            })
    swaps.sort(key=lambda x: x["d_rouge"])
    for i, x in enumerate(swaps):
        x["rank"] = i + 1
        x["priority"] = i < N_PRIORITY

    g4 = []
    with open(p("results/ground_diag_A_G4.csv"), encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q = int(r["q_idx"])
            ci = cands.get(q, {})
            g4.append({
                "id": f"G-{q}-{r['p_idx']}", "task": "G", "rank": i + 1, "priority": True, "q_idx": q,
                "p_idx": int(r["p_idx"]), "question": r["question"], "sentence": r["sentence"],
                "passage": r["passage"], "gap": float(r["gap"]), "issup": float(r["issup_score"]),
                "issup_label": r["issup_label"], "p_ent": float(r["deberta_p_ent"]),
                "p_neu": float(r["deberta_p_neu"]), "p_con": float(r["deberta_p_con"]),
                "t5": int(r["t5_verdict"]), "cit_rec": float(r["cand_citation_rec"]),
                "short_answers": [{"q": qa["question"], "a": qa["short_answers"]} for qa in ci.get("qa_pairs", [])],
            })

    out_dir = p("results", "label_tool")
    os.makedirs(out_dir, exist_ok=True)
    data = json.dumps({"swaps": swaps, "g4": g4}, ensure_ascii=False).replace("</", "<\\/")
    html = open(os.path.join(HERE, "label_tool_template.html"), encoding="utf-8").read()
    defaults = model_labels(out_dir)
    dflt = json.dumps(defaults, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace('<script src="data.js"></script>',
                        "<script>window.LABEL_DATA = " + data + ";\nwindow.LABEL_DEFAULTS = " + dflt + ";</script>")
    write_csvs(out_dir, swaps, g4, defaults)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    n_a = sum(x["set"] == "A" for x in swaps)
    print(f"swaps with different text: {len(swaps)} (A {n_a}, B {len(swaps) - n_a}); "
          f"priority {min(N_PRIORITY, len(swaps))}; ROUGE worse on {sum(x['d_rouge'] < 0 for x in swaps)}")
    print(f"G4 rows: {len(g4)}")
    print("wrote", os.path.normpath(os.path.join(out_dir, "index.html")))


if __name__ == "__main__":
    main()
