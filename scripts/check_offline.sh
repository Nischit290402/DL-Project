#!/usr/bin/env bash
# Offline check of the merged code: re-runs every division's CPU steps from the committed files and prints the
# headline numbers next to the values reported in results/. No GPU, no cluster, no extra packages (Python 3 only).
#   bash scripts/check_offline.sh            # writes its working files to out/check/
# GPU steps (Self-RAG generation, ALCE/T5 scoring, NLI models) are not re-run here: their committed outputs are used.
set -euo pipefail
cd "$(dirname "$0")/.."
O=out/check
mkdir -p "$O"
PY=${PYTHON:-python3}
A=critique/splits/asqa_A.json; B=critique/splits/asqa_B.json; C=critique/splits/asqa_C.json
R=results/runs

section() { printf '\n========== %s ==========\n' "$1"; }
boot() { "$PY" critique/bootstrap.py "$@" | grep -v '^\* = '; }
same_choices() {  # same_choices NEW.json COMMITTED.json : compares the answer text question by question
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
new = {x.get("q_idx", i): x["output"] for i, x in enumerate(json.load(open(sys.argv[1]))["data"])}
old = {x.get("q_idx", i): x["output"] for i, x in enumerate(json.load(open(sys.argv[2]))["data"])}
same = sum(new[q] == old.get(q) for q in new)
print(f"  regenerated vs committed answers: {same} of {len(new)} identical")
if same != len(new):
    sys.exit("  MISMATCH: regenerated answers differ from the committed file")
EOF
}

section "0. Baselines: C0 (released code) vs C1 (--keep_special_tokens), 948 questions"
echo "  expected: citation recall +16.31 [+14.57, +18.08], precision +3.54 [+2.34, +4.72]"
boot --base $R/asqa_baseline_0_end.json.perq.jsonl --new $R/asqa_keeptok_0_end.json.perq.jsonl

section "1. Retrieval: cross-encoder reranked passages (R2, top 20 -> 5) vs C1, Set C (748)"
echo "  expected: str-em +1.7 [0.3, 3.1]"
"$PY" retrieval/analyze_runs.py $R/asqa_cands_C.json $R/retrieval/asqa_rrR2K20C_0_end.json

section "2. NLI grounding: D2 and D2-v2 vs released, Set C (748)"
echo "  expected: D2 citation recall +5.81, str-em -2.11; D2-v2 citation recall +4.39, str-em +1.13"
G="--f $R/asqa_cands_C.json --ground $R/cand_ground_C.jsonl --split $C --mode mix --agg mean"
"$PY" grounding/rerank_ground.py $G --param 0 --out $O/rr_released_lookup_C.json | tail -2
"$PY" grounding/rerank_ground.py $G --param 0.75 --out $O/rr_ground_C.json | tail -1
"$PY" grounding/rerank_ground.py $G --param 0.75 --rel $R/cand_qarel_C.jsonl --delta 1 --out $O/rr_ground_v2_C.json | tail -1
for f in rr_released_lookup_C rr_ground_C rr_ground_v2_C; do
  "$PY" grounding/lookup_perq.py --f $O/$f.json --metrics $R/cand_metrics_C_exact.jsonl > /dev/null
  # q471 (multi-segment) is not in the lookup table: take its row from the GPU scoring of the released answer
  # (recorded deviation, grounding/PREREG_C_deviations.md)
  "$PY" grounding/fill_multiseg.py --f $O/$f.json --perq $O/$f.json.perq.jsonl \
    --released $R/rr_released_C.json --released_perq $R/rr_released_C.json.perq.jsonl > /dev/null
done
same_choices $O/rr_ground_C.json $R/rr_ground_C.json
same_choices $O/rr_ground_v2_C.json $R/rr_ground_v2_C.json
echo "  -- D2"
boot --base $O/rr_released_lookup_C.json.perq.jsonl --new $O/rr_ground_C.json.perq.jsonl --split $C
echo "  -- D2-v2"
boot --base $O/rr_released_lookup_C.json.perq.jsonl --new $O/rr_ground_v2_C.json.perq.jsonl --split $C

section "3. Self-contradiction: penalty (lambda 0.3) vs released, Set C (748)"
echo "  expected: str-em +0.17 [-0.84, +1.13], citation recall -2.24 [-4.17, +0.01]"
"$PY" contradiction/rerank_contra.py --f $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl --split $C \
  --out $O/rr_released_C.json --lam 0 | tail -1
"$PY" contradiction/rerank_contra.py --f $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl --split $C \
  --out $O/rr_contra_C.json | tail -1
same_choices $O/rr_released_C.json $R/rr_released_C.json
same_choices $O/rr_contra_C.json $R/rr_contra_C.json
boot --base $R/rr_released_C.json.perq.jsonl --new $R/rr_contra_C.json.perq.jsonl --split $C
echo "  -- uncertainty signal (expected: str-em +8.36 [+3.98, +13.23] in the less contradictory half)"
"$PY" contradiction/uncertainty_test.py --cands $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl \
  --perq $R/rr_released_C.json.perq.jsonl --split $C --out $O/uncertainty_C.json > /dev/null
"$PY" - $O/uncertainty_C.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for m in ("str_em", "citation_rec"):
    r = d[m]
    print(f"  {m:13s} low - high disagreement: {r['low_minus_high']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
print("  H1 confirmed:", d["H1_confirmed"])
EOF

section "4. Critique: Self-RAG ranking as the paper intended vs C1, 948 questions"
echo "  expected: citation recall +2.85 [+1.05, +4.67], precision +2.97 [+1.24, +4.79], ROUGE-Lsum +0.58 [+0.11, +1.09]"
boot --base $R/critique_paper_scoring/full948_released.json.perq.jsonl \
     --new $R/critique_paper_scoring/full948_paper.json.perq.jsonl
echo "  -- released re-pick reproduces C1 (Set A+B, expected 0 changed, 200 identical)"
"$PY" critique/rerank.py --f $R/asqa_cands_0_end.json --variant first --out $O/rr_released.json | grep "changed\|identical"
echo "  -- ablation: learned weights (Person 2), Set B (expected str-em +2.68 [-1.15, +6.88])"
"$PY" critique/learn_weights.py test --score-dir $R/critique_p2 --config results/critique_p2_configs/learned_weights.json --out $O/p2
boot --base $R/rr_released.json.perq.jsonl --new $O/p2/learned_weights.perq.jsonl --split $B

section "Done"
echo "All steps ran. Compare each block with its 'expected' line; working files are in $O/."
