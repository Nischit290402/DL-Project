#!/usr/bin/env bash
# Combined system: ablation ladder and tests of combined/PREREG.md (CPU, about a minute). Run after
# cluster/run_combined_prep.sbatch has written results/runs/combined/ and the retrieval runs' .perq.jsonl files.
#   bash combined/run_ladder.sh
# Writes out/combined/: one re-ranked file per row and set, its per-question scores, the bootstrap tables, and
# out/combined/final_948.json (row R3 for all 948 questions) to score directly with --mauve.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
O=${OUT_DIR:-out/combined}; R=results/runs; D=${PREP_DIR:-results/runs/combined}
mkdir -p "$O"

# (works with macOS's bash 3.2: no associative arrays). OUT_DIR, PREP_DIR, CANDS_AB, CANDS_C override the
# inputs, e.g. to dry-run the script on the GTR candidates.
cands_of() { if [ "$1" = AB ]; then echo "${CANDS_AB:-$R/retrieval/asqa_rrR2K20_0_end.json}"; else echo "${CANDS_C:-$R/retrieval/asqa_rrR2K20C_0_end.json}"; fi; }
split_of() { if [ "$1" = AB ]; then echo critique/splits/asqa_A.json,critique/splits/asqa_B.json; else echo critique/splits/asqa_C.json; fi; }

for f in "$(cands_of AB).perq.jsonl" "$(cands_of C).perq.jsonl" $D/cand_metrics_rr_AB_exact.jsonl \
         $D/cand_metrics_rr_C_exact.jsonl $D/cand_ground_rr_AB.jsonl $D/cand_ground_rr_C.jsonl \
         $D/cand_qarel_rr_AB.jsonl $D/cand_qarel_rr_C.jsonl; do
  [ -s "$f" ] || { echo "missing $f: run cluster/run_combined_prep.sbatch first"; exit 1; }
done

rank_and_score() {  # rank_and_score NAME SET [rerank_combined.py options]
  local name=$1 S=$2; shift 2
  local cands; cands=$(cands_of "$S")
  "$PY" combined/rerank_combined.py --f "$cands" --split "$(split_of "$S")" "$@" --out "$O/${name}_$S.json" | head -1
  "$PY" grounding/lookup_perq.py --f "$O/${name}_$S.json" --metrics "$D/cand_metrics_rr_${S}_exact.jsonl" > /dev/null
  "$PY" grounding/fill_multiseg.py --f "$O/${name}_$S.json" --perq "$O/${name}_$S.json.perq.jsonl" \
    --released "$cands" --released_perq "$cands.perq.jsonl" > /dev/null
}

for S in AB C; do
  echo "== Set $S"
  G="--ground $D/cand_ground_rr_$S.jsonl --alpha 0.75 --rel $D/cand_qarel_rr_$S.jsonl --delta 1"
  rank_and_score R1_check     $S --critique released                 # must change 0 answers
  rank_and_score R2_critique  $S --critique paper
  rank_and_score R3_final     $S --critique paper    $G
  rank_and_score X_ground     $S --critique released $G              # off-ladder: reranked passages + grounding only
  cp "$(cands_of "$S").perq.jsonl" "$O/R1_retrieval_$S.json.perq.jsonl"  # R1 = retrieval run as released, GPU-scored
done
# R0 = C1 on the GTR passages, GPU-scored per question (results/runs/asqa_keeptok_0_end.json.perq.jsonl)
for S in AB C; do cp $R/asqa_keeptok_0_end.json.perq.jsonl "$O/R0_c1_$S.json.perq.jsonl"; done

# all 948 questions = Set A+B file + Set C file
for row in R0_c1 R1_retrieval R2_critique R3_final X_ground; do
  if [ "$row" = R0_c1 ]; then cp "$O/R0_c1_C.json.perq.jsonl" "$O/${row}_948.json.perq.jsonl"
  else cat "$O/${row}_AB.json.perq.jsonl" "$O/${row}_C.json.perq.jsonl" > "$O/${row}_948.json.perq.jsonl"; fi
done
"$PY" cluster/merge_shards.py "$O/R3_final_AB.json" "$O/R3_final_C.json" -o "$O/final_948.json" | tail -1

compare() {  # compare TITLE BASE NEW SPLIT_OR_EMPTY
  echo; echo "---- $1"
  "$PY" critique/bootstrap.py --base "$O/$2.json.perq.jsonl" --new "$O/$3.json.perq.jsonl" ${4:+--split "$4"} \
    --out "$O/boot_${3}_vs_${2}${4:+_C}.json" | grep -v '^\* = '
}
C=critique/splits/asqa_C.json
echo; echo "================ PRIMARY: Set C (748) ================"
compare "H1/H2: R3 final vs C1"                       R0_c1_C        R3_final_C     $C
compare "ladder: R1 +retrieval vs C1"                R0_c1_C        R1_retrieval_C $C
compare "ladder: R2 +critique vs R1"                 R1_retrieval_C R2_critique_C  $C
compare "ladder: R3 +grounding vs R2"                R2_critique_C  R3_final_C     $C
compare "off-ladder: retrieval + grounding vs C1"    R0_c1_C        X_ground_C     $C
echo; echo "================ SECONDARY: all 948 ================"
compare "R3 final vs C1 (948)"                        R0_c1_948      R3_final_948   ""
compare "R1 +retrieval vs C1 (948)"                   R0_c1_948      R1_retrieval_948 ""
compare "R2 +critique vs R1 (948)"                    R1_retrieval_948 R2_critique_948 ""
compare "R3 +grounding vs R2 (948)"                   R2_critique_948 R3_final_948  ""
echo; echo "Final answers for direct scoring with MAUVE: $O/final_948.json"
