# Critique division, Person 1: measuring tools + honest grades

Branch `critique-p1` (from `asqa-baseline`). Every GPU step is an `sbatch` job on TC2; every CPU step uses only
the Python standard library, so run it on your laptop (the login node is for installs and downloads only).
Laptop and cluster share files through git (code) and `scp` (output files).

## What each file does

| File | Task | Where |
| --- | --- | --- |
| `critique/make_splits.py` | Set A and Set B: 2 x 100 fixed question indices | laptop |
| `critique/score_per_question.py` + `cluster/run_critique_score.sbatch` | ALCE scores per question (str-em, ROUGE-Lsum, citation rec/prec, length; MAUVE once) | GPU |
| `critique/bootstrap.py` | 95% intervals, and paired difference between two runs | laptop |
| `self-rag/retrieval_lm/run_long_form_static.py` | new flags `--save_candidates` and `--indices_file` (recording only, answers unchanged) | GPU |
| `critique/nli_verdicts.py` + `cluster/run_critique_nli.sbatch` | honesty check, GPU part: T5-11B verdict for each grade the model wrote | GPU |
| `critique/honesty_check.py` | honesty check results + Fix 1 (temperature scaling) | laptop |
| `critique/rerank.py` | re-pick answers with the changed grade (Fix 1, Fix 2), no Self-RAG run | laptop |

## Step 0. Laptop: make the test sets and push (2 min)

```bash
python3 critique/make_splits.py
git add critique cluster/run_critique_score.sbatch cluster/run_critique_nli.sbatch self-rag/retrieval_lm/run_long_form_static.py
git commit -m "Critique P1: test sets, per-question scoring, honesty check"
git push -u origin critique-p1
```
Tell the team the two files `critique/splits/asqa_A.json` and `asqa_B.json` exist; everyone uses them.

## Step 1. Cluster: get the code and check disk space (one time)

```bash
cd ~/DL-Project && git fetch && git checkout critique-p1 && git pull
du -sh ~ ~/hf_cache ~/envs 2>/dev/null     # home quota is 100 GB; the citation model needs ~45 GB
```
If you have not set up the cluster yet: `bash cluster/setup_env.sh`, `bash cluster/get_data.sh`,
`bash cluster/setup_alce.sh` (README of `asqa-baseline`). All jobs below need `setup_alce.sh` done.

## Step 2. C0 and C1 answer files (no GPU, already in git)

They are committed on `asqa-baseline` (and on this branch) in `results/runs/`: `asqa_baseline_0_end.json` (C0) and
`asqa_keeptok_0_end.json` (C1), each with its ALCE `.score`. `git pull` on the cluster is enough.
Seconds per question (no logs committed): C1 40.2 min / 948 = 2.54 s on an L40S; C0 53.3 min / 948 = 3.37 s on
an A40 (`results/asqa_reproduction.md`).

What the files already show (counted on this branch): every C1 answer has exactly one support grade
(891 Fully supported, 16 Partially, 42 No support, counted in each answer's last segment; one answer has two). In 336 of 948 answers
the model keeps writing after that grade (`[Continue to Use Evidence]`), so those later sentences are never graded.

## Step 3. Per-question scores for C0 and C1 (GPU, two jobs, run one after the other)

```bash
J1=$(sbatch --parsable --export=ALL,F=results/runs/asqa_keeptok_0_end.json,FLAGS=--mauve cluster/run_critique_score.sbatch)
sbatch --dependency=afterany:$J1 --export=ALL,F=results/runs/asqa_baseline_0_end.json,FLAGS=--mauve cluster/run_critique_score.sbatch
```
Check: `results/runs/asqa_keeptok_0_end.json.summary.json` against `asqa_keeptok_0_end.json.score` next to it
(str-em 30.19, citation rec 66.64, prec 67.57, MAUVE 75.57; ROUGE-Lsum can differ by a few tenths, see the
note at the top of `score_per_question.py`). If str-em or citations differ by more than 0.1, stop and send me the log.

Then on the laptop (copy the two small `.perq.jsonl` files back first):
```bash
scp '<you>@10.96.189.12:DL-Project/results/runs/*.perq.jsonl' results/runs/   # quotes: zsh would expand * locally
python3 critique/bootstrap.py --base results/runs/asqa_keeptok_0_end.json.perq.jsonl
python3 critique/bootstrap.py --base results/runs/asqa_keeptok_0_end.json.perq.jsonl --split critique/splits/asqa_A.json
python3 critique/bootstrap.py --base results/runs/asqa_baseline_0_end.json.perq.jsonl --new results/runs/asqa_keeptok_0_end.json.perq.jsonl
```
The last line is C0 vs C1 with a 95% interval: the first entry of the results table. Commit the `.perq.jsonl`
files: the other divisions need C1's per-question scores for their own intervals.

## Step 4. Honesty check, label-only version from C1 (GPU, no Self-RAG run)

```bash
sbatch --export=ALL,F=results/runs/asqa_keeptok_0_end.json cluster/run_critique_nli.sbatch
# laptop, after copying results/runs/asqa_keeptok_0_end.json.verdicts.jsonl back:
python3 critique/honesty_check.py --verdicts results/runs/asqa_keeptok_0_end.json.verdicts.jsonl
```
Result: "when the model wrote [Fully supported], the checker agreed X% of the time". No probabilities yet.
C0 cannot be used here: its grades were stripped by the bug.

## Step 5. Save all candidates on Set A + Set B (GPU, about 10-15 min)

Skip this if your partner already made a `--save_candidates` file for these questions. `EXTRA` is set in front
of `sbatch` (not inside `--export=`) because `--export` splits on commas. Test on 5 questions first:
```bash
EXTRA="--keep_special_tokens --save_candidates" sbatch --export=ALL,START=0,END=5,TAG=candtest --time=00:30:00 cluster/run_asqa.sbatch
```
Check `out/asqa_candtest_0_5.json`: every item has `candidates` (5 entries at depth 1) with `support_probs`.
Then the real run:
```bash
EXTRA="--keep_special_tokens --save_candidates --indices_file ../../critique/splits/asqa_A.json,../../critique/splits/asqa_B.json" \
  sbatch --export=ALL,TAG=cands --time=01:00:00 cluster/run_asqa.sbatch
```
Output: `out/asqa_cands_0_end.json`. Keep the log (`logs/asqa_<jobid>.log`) for seconds per question.

## Step 6. Honesty check, full version, and Fix 1 (GPU job, then laptop)

```bash
sbatch --export=ALL,F=out/asqa_cands_0_end.json cluster/run_critique_nli.sbatch
# laptop (quotes keep zsh from expanding * on the laptop):
mkdir -p out && scp '<you>@10.96.189.12:DL-Project/out/asqa_cands_0_end.json*' out/
python3 critique/honesty_check.py --verdicts out/asqa_cands_0_end.json.verdicts.jsonl \
    --tune critique/splits/asqa_A.json --test critique/splits/asqa_B.json --out results/critique_honesty.json
```
Prints the calibration table, ECE and Brier before and after, and the temperature `T` chosen on Set A.

## Step 7. Re-pick answers: released setting, Fix 1, Fix 2 (laptop, seconds)

```bash
python3 critique/rerank.py --f out/asqa_cands_0_end.json --variant first --out out/rr_released.json
python3 critique/rerank.py --f out/asqa_cands_0_end.json --variant first --temperature <T> --out out/rr_fix1.json
python3 critique/rerank.py --f out/asqa_cands_0_end.json --variant mean --out out/rr_fix2_mean.json
python3 critique/rerank.py --f out/asqa_cands_0_end.json --variant min  --out out/rr_fix2_min.json
```
Sanity check: the first command must print `answer changed vs the original run: 0` and identical text for all.
The last line of each run says how many candidates have more than one grade. C1 suggests it will be close to 0
(one grade per answer, Step 2); then Fix 2 cannot change any choice, the `mean`/`min` runs will match
`rr_released`, and Fix 2 is reported as "not applicable" with that count and the 336 ungraded answers.

## Step 8. Score the re-picked files and compare on Set B (GPU, then laptop)

```bash
scp out/rr_*.json <you>@10.96.189.12:DL-Project/out/
# cluster, chained so each job starts when the previous one ends (one GPU per user):
prev=""; for f in rr_released rr_fix1 rr_fix2_mean rr_fix2_min; do
  prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --export=ALL,F=out/$f.json cluster/run_critique_score.sbatch)
done
# laptop:
scp '<you>@10.96.189.12:DL-Project/out/rr_*.perq.jsonl' out/
python3 critique/bootstrap.py --base out/rr_released.json.perq.jsonl --new out/rr_fix1.json.perq.jsonl --split critique/splits/asqa_B.json
python3 critique/bootstrap.py --base out/rr_released.json.perq.jsonl --new out/rr_fix2_mean.json.perq.jsonl --split critique/splits/asqa_B.json
python3 critique/bootstrap.py --base out/rr_released.json.perq.jsonl --new out/rr_fix2_min.json.perq.jsonl --split critique/splits/asqa_B.json
```
If the queue refuses more than 2 waiting jobs, submit the last two after the first two finish.
A fix "wins" only if the difference interval on Set B is above 0 (marked `*`). If one wins, it is run on all
948 questions later; otherwise it stays in the report as a negative result with one line of analysis.

## What to send back after each step

The job log (`logs/crit_*_<jobid>.log` or `logs/asqa_<jobid>.log`) and the printed table. Errors: the `.err` file.
