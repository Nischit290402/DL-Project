#!/usr/bin/env bash
# Recomputes every table of the report from the committed run outputs. CPU only, Python 3 standard library,
# about 2 minutes. GPU steps (Self-RAG generation, ALCE scoring, NLI models) are not re-run here: their
# committed outputs in results/runs/ are used. The README lists the cluster commands that produced them.
#   bash scripts/reproduce_report.sh          # working files go to out/reproduce/
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}; O=out/reproduce; R=results/runs; S=$R/combined/second
A=critique/splits/asqa_A.json; B=critique/splits/asqa_B.json; C=critique/splits/asqa_C.json
mkdir -p "$O"
section() { printf '\n\n################ %s ################\n' "$1"; }
same_answers() {  # same_answers NEW.json COMMITTED.json
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["data"]; b = json.load(open(sys.argv[2]))["data"]
same = sum(x["output"] == y["output"] for x, y in zip(a, b))
print(f"  regenerated answers identical to the committed file: {same} of {len(b)}")
if len(a) != len(b) or same != len(b): sys.exit("  MISMATCH")
PYEOF
}

section "Table 1. Reproduction, all 948 questions (paper: 30.0 / 35.7 / 74.3 / 66.97 / 67.82)"
echo "  expected  C0 30.13 35.83 76.09 50.22 63.92   C1 30.19 35.75 75.57 66.64 67.57"
"$PY" - <<'PYEOF'
import json
for name, f in (("C0 released code", "asqa_baseline_0_end"), ("C1 tokens kept", "asqa_keeptok_0_end")):
    s = json.load(open(f"results/runs/{f}.json.score"))
    print(f"  {name:17s}" + "  ".join(f"{k} {s[k]:.2f}" for k in ("str_em", "rougeLsum", "mauve", "citation_rec", "citation_prec")))
PYEOF

section "Table 2. Combined system vs C1, Set C (748 held-out questions)"
echo "  expected  STR-EM +2.84 [+1.10, +4.65]  citation recall +11.44 [+8.47, +14.41]  precision +11.34 [+8.57, +14.52]"
echo "            ROUGE-L +1.00 [+0.10, +1.83]  length +1.25 [-0.20, +2.78]"
OUT_DIR=$O/combined bash combined/run_ladder.sh > $O/ladder.log
same_answers $O/combined/final_948.json $R/combined/final_948.json
sed -n '/H1\/H2: R3 final vs C1/,/ladder: R1/p' $O/ladder.log | grep -v "ladder: R1"
echo "  -- all 948 questions, scored directly on the GPU (expected 33.69 / 36.82 / 76.16 / 77.49 / 78.29, 31.25 words)"
"$PY" -c "
import json; s=json.load(open('$R/combined/final_948.json.summary.json'))
print('  combined system  '+'  '.join(f'{k} {s[k]:.2f}' for k in ('str_em','rougeLsum','mauve','citation_rec','citation_prec','length')))"

section "Tables 3 and 4. Ablation over all eight combinations of the three components, Set C"
echo "  expected  full system 32.87 / 36.85 / 77.38 / 78.25; reranking alone +1.71 [+0.35, +3.16]; removed 1.48 and 5.69"
"$PY" combined/factorial.py $O/factorial

section "Results paragraph: which restored score term matters (Sets A and B, 200 questions)"
echo "  expected  picks changed: utility only 7, likelihood only 65, both 66; citation recall -0.25 / +3.40 / +3.65"
mkdir -p $O/critique_paper_scoring
"$PY" critique/paper_scoring_check.py $O/critique_paper_scoring
echo "  -- full critique score vs C1 on all 948 (expected citation recall +2.85 [+1.05, +4.67])"
"$PY" critique/bootstrap.py --base $R/critique_paper_scoring/full948_released.json.perq.jsonl \
  --new $R/critique_paper_scoring/full948_paper.json.perq.jsonl | grep -v '^\* = '

section "Table 5. Ideas that did not pass"
echo "  -- grounding without the gate, Set C (expected STR-EM -2.11 [-3.82, -0.52])"
G="--f $R/asqa_cands_C.json --ground $R/cand_ground_C.jsonl --split $C --mode mix --agg mean"
for v in "released_lookup 0" "ground 0.75"; do set -- $v
  "$PY" grounding/rerank_ground.py $G --param $2 --out $O/rr_$1_C.json > /dev/null
  "$PY" grounding/lookup_perq.py --f $O/rr_$1_C.json --metrics $R/cand_metrics_C_exact.jsonl > /dev/null
  "$PY" grounding/fill_multiseg.py --f $O/rr_$1_C.json --perq $O/rr_$1_C.json.perq.jsonl \
    --released $R/rr_released_C.json --released_perq $R/rr_released_C.json.perq.jsonl > /dev/null
done
"$PY" critique/bootstrap.py --base $O/rr_released_lookup_C.json.perq.jsonl --new $O/rr_ground_C.json.perq.jsonl --split $C | grep -v '^\* = '
echo "  -- contradiction penalty, Set C (expected citation recall -2.24 [-4.17, +0.01])"
"$PY" critique/bootstrap.py --base $R/rr_released_C.json.perq.jsonl --new $R/rr_contra_C.json.perq.jsonl --split $C | grep -v '^\* = '
echo "  -- learned critique weights, Set B (expected citation recall about -6.6)"
"$PY" critique/learn_weights.py test --score-dir $R/critique_p2 --config results/critique_p2_configs/learned_weights.json --out $O/p2 > /dev/null
"$PY" critique/bootstrap.py --base $R/rr_released.json.perq.jsonl --new $O/p2/learned_weights.perq.jsonl --split $B | grep -v '^\* = '
echo "  -- 10 or 20 GTR passages, Sets A and B (expected STR-EM -0.2 and -1.0; the oracle row is the +9.2 of the Diagnosis)"
"$PY" retrieval/analyze_runs.py $R/asqa_cands_0_end.json $R/retrieval/asqa_oracle5_0_end.json $R/retrieval/asqa_nd10_0_end.json $R/retrieval/asqa_nd20_0_end.json \
  | grep -A3 '^=== ' | grep '^=== \|answer str-em'
echo "  -- temperature calibration (Set B STR-EM -1.75): numbers and procedure in results/critique_p1.md and results/critique_p2.md"

section "Table 6. A second segment (exploratory), Set C"
echo "  expected  new passage, checked +5.13 [+4.14, +6.24], citation recall +0.01; own choice +6.74, -9.72; same passage, checked +1.64"
SEG="$S/seg2_AB.json,$S/seg2_BC.json $S/seg2_AB.ground.jsonl,$S/seg2_BC.ground.jsonl $S/seg2_AB.ground_ev.jsonl,$S/seg2_BC.ground_ev.jsonl"
SAME="$S/seg2_SAME.json $S/seg2_SAME.ground.jsonl $S/seg2_SAME.ground_ev.jsonl"
"$PY" retrieval/second_segment_pick.py $R/combined/final_948.json $O/seg2_filter_948.json filter 0.5 $SEG > /dev/null
"$PY" retrieval/second_segment_pick.py $R/combined/final_948.json $O/seg2_beam_948.json beam 0.5 $SEG > /dev/null
"$PY" retrieval/second_segment_pick.py $R/combined/final_948.json $O/seg2_same_filter_948.json filter 0.5 $SAME > /dev/null
"$PY" retrieval/second_segment_pick.py $R/combined/final_948.json $O/seg2_same_beam_948.json beam 0.5 $SAME > /dev/null
for v in seg2_filter_948 seg2_beam_948 seg2_same_filter_948 seg2_same_beam_948; do same_answers $O/$v.json $S/$v.json; done
"$PY" scripts/second_segment_table.py

section "Done"
echo "Compare each block with its 'expected' line and with the report. Working files are in $O/."
