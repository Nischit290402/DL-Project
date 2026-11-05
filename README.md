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

## Known upstream bug (fix before running PopQA)
`self-rag/retrieval_lm/run_short_form.py`, in `main()`, passes `max_depth=args.max_depth` to
`call_model_rerank_w_scores_batch(...)`, which has no such parameter. Every run fails with
`TypeError: ... unexpected keyword argument 'max_depth'`. The argument is unused in short-form inference, so
remove `max_depth=args.max_depth, ` from that call (one line, around line 315).

## Why the old pins
`run_short_form.py` asks vLLM for full-vocabulary logprobs (`logprobs=32016`) and reads them as floats.
vLLM >= 0.4 caps logprobs and returns `Logprob` objects, so the upstream code only runs unmodified on the
vllm 0.2.x line. Upgrading is a later task once we modify the inference loop.
