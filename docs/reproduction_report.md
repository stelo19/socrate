# TutorMoments — Reproduction Report

EURECOM semester project · Stefano Longo, Daniele Priola · last updated 2 Nov 2026

## Versions

| Item | Value |
|---|---|
| Code | allenai/tutormoments @ `c6e26d7` (19 Oct 2026) |
| Dataset | allenai/tutormoments-preview @ `3bb0c104d65e29facffca7ba864ef10fa22d0e78` |
| Environment | WSL Ubuntu, Python venv, `pip install -e ".[dev,analysis]"` |
| Test suite | `pytest tests/tutormoments -q` → 766 passed, 13 skipped |
| Script | `recompute_table8.py` (uses the repo's own `tutormoments.report.aggregate`) |

## Experiment 1 — Table 8 recomputed from released replays (no API calls)

**Method.** For each of the 7 tutor models × 2 prompt modes, read the judge's annotation of each released replay in `benchmark_520` (`annotation.annotations[0]`: `action_label`, `overscaffold_decomposed`), join the gold label from `moments.dimension`, and compute Appropriate Scaffolding (S), Appropriate Rigor (R) and Avoids Over-scaffolding (O) with the official aggregation rule. Two populations: all 520 moments per cell, and the 100-moment subset (`source_round` ∈ {s42, s43, topup}). 95% confidence intervals from a bootstrap that resamples whole tutoring sessions (1,000 resamples, 112 sessions).

**Data checks.** 520 gold labels (260 scaffolding / 260 rigor). 7,280 replays, none with failed scoring. `source_round`: s42 700, s43 546, topup 154, round2 5,824, topup520 56. Paper subset = 50 scaffolding + 50 rigor per cell.

**Result: exact replication.** On all 520 moments, all 42 values match paper Table 8 to three decimals. The 100-moment subset does not match (differences up to ±0.12), so Table 8 uses all 520 moments, as the paper states. The 100-moment cells are only behind Table 9 (KL) and Figures 4–6.

| Model | Prompt | S [95% CI] | R [95% CI] | O [95% CI] | Matches paper |
|---|---|---|---|---|---|
| Claude Opus 4.8 | plain | 0.615 [0.54, 0.68] | 0.208 [0.14, 0.27] | 0.462 [0.41, 0.51] | yes |
| Claude Sonnet 4.6 | plain | 0.573 [0.52, 0.63] | 0.085 [0.05, 0.12] | 0.437 [0.39, 0.49] | yes |
| DeepSeek V4 Pro | plain | 0.485 [0.42, 0.55] | 0.088 [0.06, 0.13] | 0.379 [0.33, 0.43] | yes |
| Gemini 2.5 Pro | plain | 0.727 [0.67, 0.78] | 0.223 [0.16, 0.28] | 0.546 [0.50, 0.60] | yes |
| Gemini 3.5 Flash | plain | 0.723 [0.66, 0.78] | 0.158 [0.11, 0.20] | 0.598 [0.55, 0.65] | yes |
| GPT 5.5 | plain | 0.269 [0.20, 0.33] | 0.046 [0.02, 0.08] | 0.173 [0.14, 0.21] | yes |
| GPT 5.4 mini | plain | 0.181 [0.12, 0.24] | 0.023 [0.00, 0.05] | 0.188 [0.15, 0.23] | yes |
| Claude Opus 4.8 | eval-aware | 0.858 [0.81, 0.90] | 0.831 [0.77, 0.88] | 0.896 [0.86, 0.92] | yes |
| Claude Sonnet 4.6 | eval-aware | 0.831 [0.79, 0.87] | 0.792 [0.74, 0.84] | 0.913 [0.89, 0.94] | yes |
| DeepSeek V4 Pro | eval-aware | 0.712 [0.66, 0.76] | 0.492 [0.42, 0.56] | 0.623 [0.57, 0.67] | yes |
| Gemini 2.5 Pro | eval-aware | 0.896 [0.85, 0.93] | 0.712 [0.65, 0.77] | 0.854 [0.82, 0.89] | yes |
| Gemini 3.5 Flash | eval-aware | 0.885 [0.85, 0.92] | 0.646 [0.58, 0.71] | 0.817 [0.78, 0.85] | yes |
| GPT 5.5 | eval-aware | 0.773 [0.72, 0.83] | 0.531 [0.45, 0.61] | 0.665 [0.61, 0.72] | yes |
| GPT 5.4 mini | eval-aware | 0.738 [0.68, 0.80] | 0.404 [0.33, 0.48] | 0.598 [0.55, 0.65] | yes |

**Classification:** exact replication of the *aggregation* step. Note what this does and does not show: it confirms that the published judgments produce the published numbers. It does not re-run the tutors, the simulated student or the judge, so it says nothing yet about whether those LLM steps are stable (that needs new replays, Experiment 2).

## What the confidence intervals add (not reported in the paper)

Interval widths range from 0.04 to 0.16. Checked by whether 95% intervals overlap — a conservative check; a paired test is still to do.

- **Prompt effect is robust.** For every model and every metric, the evaluation-aware interval lies entirely above the plain one.
- **"Known orderings" are only partly supported.** The paper says the benchmark recovers Opus 4.8 > Sonnet 4.6 and GPT-5.5 > GPT-5.4 mini (§5).
  - Opus 4.8 vs Sonnet 4.6: separated only on plain-prompt R; overlapping on the other five comparisons.
  - GPT-5.5 vs GPT-5.4 mini: overlapping on all six comparisons. With the plain prompt, GPT-5.4 mini even has a slightly higher O.
- **Top of the eval-aware leaderboard is not separable.** Opus 4.8, Sonnet 4.6 and Gemini 2.5 Pro overlap on S and O; on R, Opus 4.8 is separated from Gemini 2.5 Pro but not from Sonnet 4.6.
- Of the 21 model pairs per metric, 9–15 are separated, so the benchmark does distinguish broad tiers (e.g. GPT models vs the rest on plain O), but not every rank.

**Caveat.** Overlap is a conservative test: all models are evaluated on the same 520 moments, so a **paired** bootstrap of the differences would have more power. Some pairs may separate under the paired test. That is the next free analysis.

## Still to do

1. Paired bootstrap of model differences (free, same data).
2. Experiment 2: small real run (`--sample 10`, one Claude tutor) — needs API budget.
3. Stress tests, to agree with the supervisor.
