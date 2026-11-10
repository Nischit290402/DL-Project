# DL-Project: Self-RAG on the NTU CCDS GPU cluster

AI6103 group project (Route B). Baseline: [SELF-RAG](https://arxiv.org/abs/2310.11511), code copied from
[AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) at commit `1fcdc420e48f50a7d7ab1ece5494221b93252e99`
(2024-03-19, MIT license) into `self-rag/`.

## Layout
```
self-rag/                upstream code
cluster/setup_env.sh     build conda env ~/envs/selfrag (py3.10, vllm 0.2.6, torch 2.1.2 cu121)
cluster/env.sh           activate env + HF_HOME; sourced by every job
cluster/get_data.sh      download selfrag_llama2_7b (HF) + eval_data (HF mirror)
cluster/smoke_test.py    README quick-start prompts, checks reflection tokens
cluster/run_smoke.sbatch 1 GPU, 30 min
cluster/run_popqa.sbatch 1 GPU, 4 h: PopQA adaptive retrieval, ndocs 10
cluster/requirements-lock.txt  pip freeze of the working env
cluster/sync.ps1         laptop -> cluster code sync (PowerShell, VPN on); git clone on the cluster is simpler
results/                 baseline numbers
data/ out/ logs/         gitignored
```

## Getting started on this branch

Cluster: login `10.96.189.12` (NTU VPN on, no GPU on the login node), partition `MGPU-TC2`, QoS `normal` =
1 GPU (A40 or L40S, 46 GB), 10 CPU, 30 GB RAM, 6 h wall, 2 jobs max. The login node has internet.

One-time setup (about 15 min):
```bash
git clone -b cluster-setup https://github.com/Nischit290402/DL-Project.git ~/DL-Project
cd ~/DL-Project
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
git checkout -b <your-branch> cluster-setup
# edit code, commit, push, then on the cluster:
cd ~/DL-Project && git fetch && git checkout <your-branch> && git pull
sbatch cluster/run_smoke.sbatch    # must still print SMOKE_OK
sbatch cluster/run_popqa.sbatch    # about 40 min; logs/popqa_<jobid>.log ends with "Final result: <acc>"
```
Compare `Final result` against `results/popqa_baseline.md` (0.5497 at this commit; paper 54.9). A change that
does not touch inference should reproduce that number exactly (greedy decoding, temperature 0). Log every
experiment as an ablation: idea, hypothesis, setup, result, decision.

## ASQA long-form baseline (branch `asqa-baseline`)

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

## Known upstream bug (already fixed on this branch, commit 90616df)
`self-rag/retrieval_lm/run_short_form.py`, in `main()`, passes `max_depth=args.max_depth` to
`call_model_rerank_w_scores_batch(...)`, which has no such parameter. Every run fails with
`TypeError: ... unexpected keyword argument 'max_depth'`. The argument is unused in short-form inference, so
remove `max_depth=args.max_depth, ` from that call (one line, around line 315).

## Why the old pins
`run_short_form.py` asks vLLM for full-vocabulary logprobs (`logprobs=32016`) and reads them as floats.
vLLM >= 0.4 caps logprobs and returns `Logprob` objects, so the upstream code only runs unmodified on the
vllm 0.2.x line. Upgrading is a later task once we modify the inference loop.
