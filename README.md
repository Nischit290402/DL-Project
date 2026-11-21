# DL-Project: improving Self-RAG on ASQA long-form QA

AI6103 group project (Route B). Baseline: [SELF-RAG](https://arxiv.org/abs/2310.11511), code copied from
[AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) at commit `1fcdc420e48f50a7d7ab1ece5494221b93252e99`
(2024-03-19, MIT license) into `self-rag/`.

This `main` branch merges the four division branches unchanged: `retrieval`, `nli-grounding` (which contains
`self-contradiction`) and `critique-final`. All of them start from `asqa-baseline` + `critique-p1`, so they share
one Self-RAG inference script, one evaluation protocol and the same test sets.

## Shared setup

- Model and data: Self-RAG 7B (`selfrag/selfrag_llama2_7b`), ASQA dev (948 questions, ALCE GTR top-100 passages),
  upstream README long-form settings (`cluster/run_asqa.sbatch`), NVIDIA L40S only.
- Baselines: **C0** = released code (`results/runs/asqa_baseline_0_end.json`); **C1** = C0 + `--keep_special_tokens`
  (`results/runs/asqa_keeptok_0_end.json`). Every experiment is compared with C1.
- Test sets (`critique/splits/`): Set A (100, tuning), Set B (100, first test), Set C (748, the remaining questions,
  held out for pre-registered confirmation).
- Metrics: ALCE (commit `246c476`) str-em, ROUGE-Lsum, MAUVE, citation recall/precision (T5-11B NLI), plus length,
  seconds per question and contradiction rate. Intervals: paired bootstrap over questions.

## Results by division (each against C1's choices on the same questions)

| Division | Change | Test set | Main result | Write-up |
| --- | --- | --- | --- | --- |
| Baseline | `--keep_special_tokens` (C0 -> C1) | all 948 | citation recall +16.31 [+14.57, +18.08] | `results/asqa_reproduction.md`, `results/critique_p1.md` |
| 1 Retrieval | cross-encoder reranks GTR top 20, keeps 5 | Set C (748) | str-em +1.71 [+0.31, +3.14] | `results/retrieval.md` |
| 2 NLI grounding | D2-v2: relevance-gated NLI grounding term in the ranking | Set C (748) | str-em +1.13 [+0.01, +2.28], citation recall +4.39 [+2.31, +6.34] | `results/nli_grounding_setC.md` |
| 3 Critique | ranking with the paper's full critique score (utility + sequence score restored) | all 948 | citation recall +2.85 [+1.05, +4.67], precision +2.97 [+1.24, +4.79] | `results/critique_paper_scoring.md` |
| 4 Self-contradiction | penalty for candidates the others contradict | Set C (748) | no str-em gain (+0.17 [-0.84, +1.13]); disagreement confirmed as an uncertainty signal | `results/self_contradiction.md` |

Each change was tested on its own; no configuration combining them has been run.

## Layout
```
self-rag/                upstream code; retrieval_lm/run_long_form_static.py has the team's flags
                         (--keep_special_tokens, --save_candidates, --indices_file, --start/--end, ...)
cluster/                 environment setup, data download and every sbatch job (all GPU work)
critique/                Division 3 + shared tools: splits, per-question ALCE scoring, bootstrap, re-ranking
retrieval/               Division 1: passage coverage, cross-encoder reranking, run analysis
grounding/               Division 2: NLI grounding scores, re-ranking, tuning, label tool
contradiction/           Division 4: candidate NLI, contradiction penalty, uncertainty test
scripts/check_offline.sh re-runs every division's CPU steps from committed files (no GPU)
results/                 write-ups (*.md) and committed run outputs (results/runs/)
data/ out/ logs/         gitignored
```

## Testing the merged code

1. Offline, any laptop with Python 3, no extra packages (about 2 minutes):
   ```bash
   bash scripts/check_offline.sh
   ```
   It re-picks answers with each division's code from the committed candidates and scores, and prints every
   headline number next to its expected value. Regenerated answers must match the committed files exactly.
2. On the cluster (GPU), see "Cluster setup" below and each division's README for the full pipeline.

## Cluster setup

Cluster: login `10.96.189.12` (NTU VPN on, no GPU on the login node), partition `MGPU-TC2`, QoS `normal` =
1 GPU (A40 or L40S, 46 GB), 10 CPU, 30 GB RAM, 6 h wall, 2 jobs max. The login node has internet.

One-time setup (about 15 min):
```bash
git clone https://github.com/Nischit290402/DL-Project.git ~/DL-Project
cd ~/DL-Project
mkdir -p logs out
# if conda asks to accept Anaconda's Terms of Service, create the env from conda-forge first:
#   module load miniconda3/25.5.1 && eval "$(conda shell.bash hook)"
#   conda create -y -p ~/envs/selfrag --override-channels -c conda-forge python=3.10
bash cluster/setup_env.sh          # conda env at ~/envs/selfrag
bash cluster/get_data.sh           # 13 GB model into ~/hf_cache, eval data into data/eval_data
```

### 1. Check it works
```bash
sbatch cluster/run_smoke.sbatch
squeue -u $USER                    # wait until the job leaves the queue, about 2 min
tail -5 logs/smoke_<jobid>.log     # last line must be SMOKE_OK
```
Expected outputs in the log (same as the upstream README):
```
The capital of France is Paris.[Utility:5]</s>
[Relevant]Alpacas are considerably smaller than llamas.[Fully supported][Utility:5]</s>
```

### 2. Check your changes on a new branch
```bash
git checkout -b <your-branch> main
# edit code, commit, push, then on the cluster:
cd ~/DL-Project && git fetch && git checkout <your-branch> && git pull
sbatch cluster/run_smoke.sbatch    # must still print SMOKE_OK
sbatch cluster/run_popqa.sbatch    # about 40 min; logs/popqa_<jobid>.log ends with "Final result: <acc>"
```
Compare `Final result` against `results/popqa_baseline.md` (0.5497 at this commit; paper 54.9). A change that
does not touch inference should reproduce that number exactly (greedy decoding, temperature 0). Log every
experiment as an ablation: idea, hypothesis, setup, result, decision.

## ASQA long-form baseline (from branch `asqa-baseline`)

Needed for the self-contradiction experiment: short-form PopQA produces one segment per answer, so cross-segment
contradiction only exists in long-form beam search (`call_model_beam_batch` in `run_long_form_static.py`).

Changes to `run_long_form_static.py` (outputs identical to upstream at default settings):
- `--start/--end` to shard the 948 questions across jobs (6 h wall limit), plus a s/question progress line.
- `--w_rel/--w_sup/--w_use` are now actually used. Upstream ignored them and hardcoded 1.0/1.0/0.5, so the
  `--w_use` default is changed to 0.5 to keep the baseline unchanged.
- `--gpu_memory_utilization` (default 0.9, the vLLM default) to leave room for an extra NLI model later.

Cluster limits that shape the plan (TC2 user guide, QoS `normal`): the limits are per user and summed over all
your jobs: 1 GPU, 10 CPU, 30 GB RAM, 2 jobs, 6 h each. Two GPU jobs from one account therefore cannot run at the
same time; chain them with `--dependency`, or run the shards from two different members' accounts. Home quota is
100 GB and things break above 98%; check with `du -sh ~/*` or `ncdu`. Do not run Python on the login node except
installs and downloads; everything else goes through `sbatch`.

Steps:
```bash
ls data/eval_data | grep asqa                       # need asqa_eval_gtr_top100.json from get_data.sh
# 1. pilot (30 questions): read the s/question lines in the log to size the shards
sbatch --export=ALL,START=0,END=30,TAG=pilot --time=01:00:00 cluster/run_asqa.sbatch
# 2. full baseline in two shards, the second starts when the first ends (one GPU per user)
JOB_A=$(sbatch --parsable --export=ALL,START=0,END=474,TAG=baseline cluster/run_asqa.sbatch)
sbatch --dependency=afterany:$JOB_A --export=ALL,START=474,TAG=baseline cluster/run_asqa.sbatch
python cluster/merge_shards.py out/asqa_baseline_0_474.json out/asqa_baseline_474_end.json -o out/asqa_baseline.json
# 3. scoring setup (login node, one time, after setup_env.sh). Adds scoring packages to the selfrag env and
#    downloads scorer models; the T5-11B citation model is ~45 GB (SKIP_T5=1 to leave it out)
bash cluster/setup_alce.sh
sbatch --export=ALL,F=out/asqa_baseline.json cluster/run_alce_eval.sbatch     # writes out/asqa_baseline.json.score
# 4. go/no-go: how often do baseline answers contradict themselves across segments?
sbatch --export=ALL,F=out/asqa_baseline.json,TAG=baseline cluster/run_contra.sbatch
```
The full 948 questions take 40-55 min on one L40S, so one job is enough
(`sbatch --export=ALL,START=0,TAG=baseline --time=03:00:00 cluster/run_asqa.sbatch`); sharding is only a fallback.

**GPU model matters.** TC2N01/N02 have A40s (16 GPUs) and TC2N03-N08 have L40S (24 GPUs); running the same
job on the two models changes about 3% of answers. `run_asqa.sbatch` and `run_alce_eval.sbatch` exclude N01/N02 and
stop with `WRONG_GPU` if they still land on a non-L40S node (`REQUIRE_GPU=""` turns the check off).

**Use `--keep_special_tokens` for every ASQA experiment** (`EXTRA=--keep_special_tokens`). Without it vLLM strips
Self-RAG's reflection tokens, which disables segment expansion, the retrieval threshold and `--ignore_cont`, and
lowers citation recall from 66.6 to 50.2. Details and numbers: `results/asqa_reproduction.md`.
Expected numbers: the paper (Table 2, 7B) reports str-em 30.0, rouge 35.7, MAUVE 74.3, citation 66.9/67.8.
Upstream issue #4 (still open) shows the released code does not reproduce the citation metrics: independent
7B reruns get about str-em 30.2, rouge 35.8, MAUVE 68-76, citation recall ~50, precision ~64. So compare every
experiment against OUR reproduced baseline, not the paper, and say so in the report.
The author's own 7B ASQA predictions are linked in that issue
(https://drive.google.com/file/d/190YHd-7fCzZ7ml31WuCKwaFpHqBl6UC0/view); `contradiction_rate.py` can run on
them before our baseline finishes (it falls back to splitting the cited output when `intermediate` is missing).

## Known upstream bug (fixed, commit 90616df)
`self-rag/retrieval_lm/run_short_form.py`, in `main()`, passes `max_depth=args.max_depth` to
`call_model_rerank_w_scores_batch(...)`, which has no such parameter. Every run fails with
`TypeError: ... unexpected keyword argument 'max_depth'`. The argument is unused in short-form inference, so
remove `max_depth=args.max_depth, ` from that call (one line, around line 315).

## Why the old pins
`run_short_form.py` asks vLLM for full-vocabulary logprobs (`logprobs=32016`) and reads them as floats.
vLLM >= 0.4 caps logprobs and returns `Logprob` objects, so the upstream code only runs unmodified on the
vllm 0.2.x line. Upgrading is a later task once we modify the inference loop.
