# Handoff to Person 2 (critique division): learned weights

Person 1 (baselines + honesty of the self-grades) is finished; results are in `results/critique_p1.md`.
This note lists what is already built for Person 2, the two scoring problems found on the way, and what is left.
Start your work on a new branch from this one:

```bash
git fetch origin && git checkout -b critique-p2 origin/critique-p1
```

## Your task in one paragraph

Self-RAG writes 5 candidate answers per ASQA question (one per retrieved passage) and keeps the one with the
highest score `w_rel * relevance + w_sup * support + w_use * utility` (+ `exp(seq_score)` with `--use_seqscore`),
with hand-set weights 1.0 / 1.0 / 0.5. Your job: find weights that pick better answers, choosing them on Set A
and proving them on Set B.

## Already done: do not rebuild these

The task split listed the first two items under Person 2; Person 1 built them while doing the honesty check.

| Item | Where |
| --- | --- |
| Save every candidate with its raw grade probabilities | `--save_candidates` in `self-rag/retrieval_lm/run_long_form_static.py` (recording only; answers unchanged) |
| Run only chosen questions | `--indices_file critique/splits/asqa_A.json,critique/splits/asqa_B.json` (same script) |
| Saved candidates for Set A + Set B (200 questions x 5) | `out/asqa_cands_0_end.json`, 2.9 MB, **not in git**: on Person 1's cluster account (`~saksham010/DL-Project/out/`) and laptop. Ask Person 1 for a copy, or regenerate in about 8 min (command below) |
| Test sets | `critique/splits/asqa_A.json` (tune), `asqa_B.json` (test), 100 questions each |
| Per-question scoring (ALCE metrics) | `critique/score_per_question.py` + `cluster/run_critique_score.sbatch` |
| 95% paired bootstrap | `critique/bootstrap.py` (`--split` limits to a test set) |
| Re-pick answers offline from the saved candidates | `critique/rerank.py` (changes only the support term today; extend it for your weights) |
| Baseline on these 200 questions | `out/rr_released.json` = C1's answers, reproduced 200/200 from the saved candidates, scored in `out/rr_released.json.perq.jsonl` (not in git, Person 1's laptop and cluster). The same answers' C1 scores are in git: `results/runs/asqa_keeptok_0_end.json.perq.jsonl` |

Regenerate the saved candidates (cluster, L40S, about 8 min):
```bash
EXTRA="--keep_special_tokens --save_candidates --indices_file ../../critique/splits/asqa_A.json,../../critique/splits/asqa_B.json" \
  sbatch --export=ALL,TAG=cands --time=01:00:00 cluster/run_asqa.sbatch
```

## Two scoring problems in the released code (important for weight tuning)

Both are in upstream Self-RAG too, and both are left as released so that C1 stays the paper's setting.

1. **Utility is always 0, so `--w_use` does nothing.** In `run_step_generation_batch`
   (`run_long_form_static.py`, around line 86) the utility probabilities are read by looping over `grd_tokens`
   (the 3 support tokens) instead of `ut_tokens` (the 5 utility tokens). The dict gets 3 entries, the
   `len(...) == 5` check fails, and `utility_score` falls back to 0.0.
2. **`--use_seqscore` never reaches the scorer.** `call_model_beam_batch` calls `run_step_generation_batch`
   (around line 261) without `use_seqscore`, so it stays False and `exp(seq_score)` is never added.

So the ranking actually used today is **relevance + support** only. Tuning `w_use` through the script changes
nothing. The saved candidates already hold what you need to test the full score offline, without re-running
Self-RAG:

| Field per candidate | Meaning |
| --- | --- |
| `relevance_score`, `ground_score` | the two terms the released ranking uses |
| `final_score` | the released score (= relevance + support with the default weights) |
| `utility_probs` | the 5 utility-token probabilities, read correctly from `ut_tokens` |
| `seq_score` | average token log-probability of the candidate |
| `support_probs` | the 3 support-token probabilities (one entry; 25 of 1000 candidates have none) |
| `p_idx`, `ctx`, `pred`, `processed_pred` | passage index, passage, raw and processed generation |

Utility as the code intends it: `sum(w_i * p_i) / sum(p_i)` with w = (-1, -0.5, 0, 0.5, 1) for Utility:1..5.

## What is left for you

1. **Score all 1000 candidates once.** Write 5 files, the k-th picking passage k for every question (adapt
   `critique/rerank.py`), and score each with `cluster/run_critique_score.sbatch` (about 2-3 min per file, one
   GPU job at a time). Afterwards any weight setting is judged by looking up the chosen candidate's scores: no GPU.
2. **Best-possible check (oracle):** per question, the best of the 5 candidates. This is the ceiling for any
   weighting; if it is close to C1, say so early.
3. **Grid search on Set A** over w_rel, w_sup, w_use (with the corrected utility) and on/off for seq_score.
4. **Test the best weights on Set B**, paired against `rr_released` with `critique/bootstrap.py --split
   critique/splits/asqa_B.json`. A win needs the difference interval above 0 (`*`). If it wins, run it on all 948
   questions with `--keep_special_tokens`.
5. **Config file per results row** (from the task split).
6. **Results summary** like `results/critique_p1.md`, negative results included with one line of analysis.

## Rules (team shared rules)

- Always `--keep_special_tokens`; compare against C1, never only C0.
- Choose on Set A, report on Set B; never tune and report on the same questions.
- Runs on L40S (the scripts pin this); differences under about 0.5 str-em are noise.
- Metrics: str-em, ROUGE-Lsum, citation recall/precision (+ MAUVE on full runs), length, seconds per question.

## Lessons from Person 1 that apply to weights

- Fix 1 (temperature scaling of the support grade) made the grades better calibrated (ECE 0.255 -> 0.121) but
  lowered str-em on Set B by 1.75 [-3.75, -0.25]: shrinking the support term changed 13 of 200 picks for the worse.
  Expect weight changes to move few answers, so effects will be small next to the Set B intervals.
- The model grades 94% of answers `[Fully supported]`; an outside checker agrees on 76% of chosen answers but only
  57% of all candidates. The support grade is weakest exactly when comparing candidates.
- On zsh (macOS), quote remote wildcards in scp: `scp 'user@host:path/*.jsonl' dir/`.
