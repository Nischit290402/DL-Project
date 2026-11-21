"""Person 2: export ALCE inputs, select weights on A, and evaluate locked weights on B."""
import argparse
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

from bootstrap import METRICS, load
from common import load_split_indices
from rerank import candidate_output, new_ground, pick_candidate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
SPLITS = ROOT / "critique" / "splits"
POSITIONS = range(5)
BASELINE = {"w_rel": 1, "w_sup": 1, "w_use": 0, "use_seqscore": False}
# Nonnegative weights. Skip the all-zero rule, which just selects the first passage.
GRID = {
    "w_rel": [0, 0.5, 1, 2],
    "w_sup": [0, 0.5, 1, 2, 3],
    "w_use": [0, 0.5, 1],
    "use_seqscore": [False, True],
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def first_step(item):
    candidates = [c for c in item["candidates"] if c["depth"] == 1]
    if sorted(c["p_idx"] for c in candidates) != list(POSITIONS):
        raise ValueError(f"Expected passage positions 0–4 for question {item['q_idx']}")
    return candidates


def features(candidate, temperature):
    first_utility = next(iter(candidate["utility_probs"]), {})
    probs = [first_utility.get(f"[Utility:{i}]", 0.0) for i in range(1, 6)]
    total = sum(probs)
    utility = sum(weight * p for weight, p in zip((-1, -0.5, 0, 0.5, 1), probs)) / total if total else 0.0
    support = candidate["ground_score"] if temperature == 1 else new_ground(candidate, "first", temperature)
    return candidate["relevance_score"], support, utility, math.exp(candidate["seq_score"])


def select(candidates, weights):
    settings = (weights["w_rel"], weights["w_sup"], weights["w_use"], weights["use_seqscore"])
    by_position = {c["p_idx"]: values for c, values in candidates}
    return pick_candidate(
        [c for c, _ in candidates],
        lambda c: sum(w * value for w, value in zip(settings, by_position[c["p_idx"]])),
    )


def average(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def oracle(scores, baseline, indices):
    result = {}
    for metric in METRICS[:-1]:  # length is descriptive, not a target to maximize
        best, reference = [], []
        for q in indices:
            values = [scores[p][q].get(metric) for p in POSITIONS]
            values = [value for value in values if value is not None]
            if values and baseline[q].get(metric) is not None:
                best.append(max(values))
                reference.append(baseline[q][metric])
        ceiling = sum(best) / len(best) if best else None
        c1 = sum(reference) / len(reference) if reference else None
        result[metric] = {"c1": c1, "oracle": ceiling, "n_questions": len(best),
                          "gap": ceiling - c1 if best else None}
    return result


def select_on_a(items, scores, baseline, indices, temperature):
    candidates = {q: [(c, features(c, temperature)) for c in first_step(items[q])] for q in indices}
    best_key, best = None, None
    for values in itertools.product(*GRID.values()):
        if not any(values):
            continue
        weights = dict(zip(GRID, values))
        chosen = {q: select(candidates[q], weights)["p_idx"] for q in indices}
        rows = [scores[p][q] for q, p in chosen.items()]
        metrics = {metric: average(rows, metric) for metric in METRICS}
        # Select on str-em. Break ties using citation recall, precision, ROUGE, then
        # prefer the rule closest to C1, measured as distance from the default weights.
        quality = [metrics[k] for k in ("str_em", "citation_rec", "citation_prec", "rougeLsum")]
        distance = sum(abs(weights[k] - BASELINE[k]) for k in GRID)
        key = tuple(v if v is not None else -math.inf for v in quality) + (-distance,)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "weights": weights, "temperature": temperature, "set_a_metrics": metrics,
                "set_a_changed_candidates": sum(p != baseline[q]["chosen_p_idx"] for q, p in chosen.items()),
            }
    return best


def export_candidates(items, out):
    for p in POSITIONS:
        data = [candidate_output(item, next(c for c in first_step(item) if c["p_idx"] == p))
                for item in items.values()]
        write_json(out / f"candidate_{p}.json", {"data": data})


def apply_config(items, config):
    started = perf_counter()
    chosen, kept = [], 0
    for item in items.values():
        path = item["best_paths"][str(item.get("output_path", 0))]
        if sum(node not in (None, 0) for node in path) != 1:
            # A first-step selector cannot substitute a whole multi-step answer.
            chosen.append({k: v for k, v in item.items() if k not in ("candidates", "best_paths")})
            kept += 1
            continue
        candidates = [(c, features(c, config["temperature"])) for c in first_step(item)]
        chosen.append(candidate_output(item, select(candidates, config["weights"])))
    return chosen, {"questions": len(chosen), "original_paths_preserved": kept,
                    "rerank_seconds_per_question": (perf_counter() - started) / len(chosen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "tune", "test", "apply"])
    parser.add_argument("--candidates", type=Path, default=RUNS / "asqa_cands_0_end.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, help="directory with candidate_0–4.json.perq.jsonl")
    parser.add_argument("--config", type=Path, help="locked configuration created by tune")
    parser.add_argument("--coverage-only", action="store_true", help="diagnostic using str-em and length only")
    args = parser.parse_args()

    items = {item["q_idx"]: item for item in read_json(args.candidates)["data"]}
    if args.action == "export":
        export_candidates(items, args.out)
        return
    if args.action == "apply":
        if args.config is None:
            parser.error("apply requires --config")
        outputs, timing = apply_config(items, read_json(args.config))
        write_json(args.out, {"data": outputs, "rerank": timing})
        return

    if args.score_dir is None:
        parser.error("tune and test require --score-dir")
    label = "A" if args.action == "tune" else "B"
    indices = load_split_indices(str(SPLITS / f"asqa_{label}.json"))
    scores = {p: load(args.score_dir / f"candidate_{p}.json.perq.jsonl") for p in POSITIONS}
    protocol = "coverage_only" if args.coverage_only else "alce"
    metrics = ["str_em", "length"] if args.coverage_only else METRICS
    for p, rows in scores.items():
        if not set(indices) <= rows.keys():
            raise ValueError(f"Passage {p} is missing questions from Set {label}")
        for q in indices:
            if any(metric not in rows[q] for metric in metrics):
                raise ValueError(f"Passage {p}, question {q}, is missing {protocol} score fields")
            if any(rows[q][metric] is None for metric in ("str_em", "length")):
                raise ValueError(f"Passage {p}, question {q}, is missing str-em or length")
    baseline_scores = load(RUNS / "rr_released.json.perq.jsonl")
    baseline_answers = {r["q_idx"]: r for r in read_json(RUNS / "rr_released.json")["data"]}
    for q in indices:
        passage = baseline_answers[q]["chosen_p_idx"]
        reference = baseline_scores[q]
        if any(not math.isclose(scores[passage][q][metric], reference[metric])
               for metric in ("str_em", "length")):
            raise ValueError(f"Scored C1 candidate does not match the reference for question {q}")
    ceilings = oracle(scores, baseline_scores, indices)

    if args.action == "tune":
        for name, temperature in (("learned_weights", 1.0), ("learned_weights_calibrated", 4.0)):
            config = select_on_a(items, scores, baseline_answers, indices, temperature)
            config.update({"selection_split": "A", "selection_metric": "str_em", "score_protocol": protocol, "grid": GRID,
                           "candidates": "asqa_cands_0_end.json", "baseline": "rr_released.json"})
            write_json(args.out / f"{name}.json", config)
        write_json(args.out / "oracle_A.json", ceilings)
        return

    if args.config is None:
        parser.error("test requires --config")
    config = read_json(args.config)
    if config.get("selection_split") != "A" or config.get("score_protocol") != protocol:
        parser.error("test requires a config selected on Set A with the same scoring protocol")
    selected, timing = apply_config({q: items[q] for q in indices}, config)
    rows = [{**scores[item["chosen_p_idx"]][item["q_idx"]], "p_idx": item["chosen_p_idx"]} for item in selected]
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.config.stem
    (args.out / f"{name}.perq.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
    )
    write_json(args.out / f"{name}.test_B.json", {
        "split": "B", "config": config, "oracle": ceilings, **timing,
        "changed_candidates": sum(row["p_idx"] != baseline_answers[row["q_idx"]]["chosen_p_idx"] for row in rows),
        "changed_answer_texts": sum(item["output"] != baseline_answers[item["q_idx"]]["output"] for item in selected),
    })


if __name__ == "__main__":
    main()
