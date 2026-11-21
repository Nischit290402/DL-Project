# ASQA reference runs

Self-RAG 7B on all 948 ASQA questions, upstream README settings (`cluster/run_asqa.sbatch`), scored with ALCE
(`cluster/run_alce_eval.sbatch`, `--citations --qa --mauve`). Full write-up: `results/asqa_reproduction.md`.

| Run | Files | Flag | Job | GPU | str-em | ROUGE-Lsum | MAUVE | Cit. rec | Cit. prec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0, released code as-is | `asqa_baseline_0_end.json` (+ `.score`) | none | 52720 | A40 | 30.13 | 35.83 | 76.09 | 50.22 | 63.92 |
| **C1, reference** | `asqa_keeptok_0_end.json` (+ `.score`) | `--keep_special_tokens` | 52783 | L40S | 30.19 | 35.75 | 75.57 | 66.64 | 67.57 |

**Compare your experiments against C1.** C0 is kept only to document the token-stripping bug.

To match C1, run on the `asqa-baseline` branch with the fix flag (two hyphens):

```bash
sbatch --export=ALL,START=0,TAG=myexp,EXTRA=--keep_special_tokens --time=03:00:00 cluster/run_asqa.sbatch
sbatch --dependency=afterany:<that job id> --export=ALL,F=out/asqa_myexp_0_end.json cluster/run_alce_eval.sbatch
```

The scripts pin jobs to L40S nodes; differences below about 0.5 str-em/ROUGE between runs are noise.

Each `.json` is `{"data": [...]}`, one item per question: `question`, `output` (answer with `[k]` citations),
`docs` (the cited passages), `intermediate` (raw generated segments), plus the ASQA fields (`qa_pairs`,
`annotations`, ...). Each `.score` is the ALCE metrics dict.
