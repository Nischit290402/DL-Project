"""Honesty check and Fix 1 (temperature scaling). CPU only, seconds. Input: nli_verdicts.py output.

Question: when Self-RAG grades its own text as supported, does an outside checker (T5-11B NLI) agree?

Part 1, label agreement (works on any verdicts file, including label-only from C1):
    for each grade the model wrote, the share of claims the checker judged supported.
Part 2, calibration (needs a --save_candidates run, i.e. grade probabilities):
    support score s = P(Fully)/sum + 0.5 * P(Partially)/sum, the same formula run_long_form_static.py uses.
    An honest grader's s matches how often the checker agrees. Reported per bin, plus
    ECE (expected calibration error: average gap between s and the agreement rate, lower is better) and
    Brier score (mean squared gap between s and the 0/1 verdict, lower is better).
Part 3, Fix 1 (temperature scaling):
    p_i -> p_i^(1/T), renormalised. T > 1 softens overconfident grades, T < 1 sharpens them.
    T is chosen on the tune split (lowest Brier) and judged on the test split, never on the same questions.

    python critique/honesty_check.py --verdicts out/asqa_cands_0_end.json.verdicts.jsonl \
        --tune critique/splits/asqa_A.json --test critique/splits/asqa_B.json --out results/critique_honesty.json

By default only the first grade of each candidate is used, because that is the one the ranking reads;
--all_claims uses every grade.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUPPORT_TOKENS, load_split_indices  # noqa: E402

FULL, PART, NONE = SUPPORT_TOKENS
T_GRID = [round(0.25 * k, 2) for k in range(1, 41)]  # 0.25 ... 10.0


def support_score(probs, temperature=1.0):
    """Self-RAG's ground_score from the three grade probabilities, optionally temperature-scaled."""
    vals = {k: probs.get(k, 0.0) ** (1.0 / temperature) if probs.get(k, 0.0) > 0 else 0.0 for k in (FULL, PART, NONE)}
    total = sum(vals.values())
    if total <= 0:
        return 0.0
    return vals[FULL] / total + 0.5 * vals[PART] / total


def brier(rows, temperature):
    return sum((support_score(r["probs"], temperature) - r["verdict"]) ** 2 for r in rows) / len(rows)


def calibration(rows, temperature=1.0, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        s = support_score(r["probs"], temperature)
        bins[min(int(s * n_bins), n_bins - 1)].append((s, r["verdict"]))
    table, ece = [], 0.0
    for b, members in enumerate(bins):
        if not members:
            continue
        mean_s = sum(s for s, _ in members) / len(members)
        agree = sum(v for _, v in members) / len(members)
        ece += len(members) / len(rows) * abs(mean_s - agree)
        table.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": len(members),
                      "model_score": round(mean_s, 3), "checker_agrees": round(agree, 3)})
    return {"ece": round(ece, 4), "brier": round(brier(rows, temperature), 4), "bins": table}


def label_agreement(rows):
    out = {}
    for label in (FULL, PART, NONE):
        sel = [r["verdict"] for r in rows if r["label"] == label]
        if sel:
            out[label] = {"n": len(sel), "checker_says_supported": round(sum(sel) / len(sel), 3)}
    return out


def print_calibration(name, cal):
    print(f"\n{name}: ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f}")
    print(f"  {'model score':>12s} {'n':>5s} {'avg model score':>16s} {'checker agrees':>15s}")
    for b in cal["bins"]:
        print(f"  {b['bin']:>12s} {b['n']:5d} {b['model_score']:16.3f} {b['checker_agrees']:15.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--tune", default="", help="split file(s) used to choose the temperature (Set A)")
    parser.add_argument("--test", default="", help="split file(s) used to report the result (Set B)")
    parser.add_argument("--all_claims", action="store_true", help="use every grade, not only the first one")
    parser.add_argument("--out", default=None, help="JSON file for all numbers (and the fitted temperature)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.verdicts) if line.strip()]
    if not args.all_claims:
        rows = [r for r in rows if r["claim_idx"] == 0]
    report = {"verdicts": args.verdicts, "claims": "all" if args.all_claims else "first grade per candidate",
              "n_rows": len(rows)}

    print("Part 1. When the model wrote this grade, how often did the checker agree the text is supported?")
    for name, subset in [("all candidates", rows), ("chosen answers only", [r for r in rows if r["selected"]])]:
        agree = label_agreement(subset)
        report[f"label_agreement_{name.replace(' ', '_')}"] = agree
        print(f"  {name}:")
        for label, v in agree.items():
            print(f"    {label:30s} n={v['n']:5d}  checker says supported: {100 * v['checker_says_supported']:.1f}%")

    prob_rows = [r for r in rows if r["probs"] is not None]
    if not prob_rows:
        print("\nNo grade probabilities in this file (label-only check). Parts 2 and 3 need a --save_candidates run.")
    else:
        tune_ids = set(load_split_indices(args.tune)) if args.tune else None
        test_ids = set(load_split_indices(args.test)) if args.test else None
        tune = [r for r in prob_rows if tune_ids is None or r["q_idx"] in tune_ids]
        test = [r for r in prob_rows if test_ids is None or r["q_idx"] in test_ids]
        print(f"\nPart 2. Calibration (tune: {len(tune)} grades, test: {len(test)} grades)")
        report["calibration_test_before"] = calibration(test)
        print_calibration("Test split, model as released (T=1)", report["calibration_test_before"])

        best_t = min(T_GRID, key=lambda t: brier(tune, t))
        report["temperature"] = best_t
        report["calibration_test_after"] = calibration(test, best_t)
        print(f"\nPart 3. Fix 1: temperature chosen on the tune split: T = {best_t}")
        print_calibration(f"Test split, temperature-scaled (T={best_t})", report["calibration_test_after"])
        if tune_ids is None or test_ids is None:
            print("warning: --tune/--test not both given, so the temperature was fitted and judged on the same data")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
