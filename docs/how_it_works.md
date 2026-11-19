# How it works

This page explains the benchmark pipeline, what each model receives, how the judge decides, how the metrics are
computed, and what our code changes. File paths under `tutormoments/` are Ai2's code (submodule, commit `c6e26d7`).

## 1. The data: moments

A **moment** is a real tutoring conversation cut at a turn chosen by expert teachers, with a gold label saying
what the tutor should do next:

- **scaffolding** — the student is stuck: give support that makes the task approachable (a guiding question,
  a hint, breaking the problem into steps) without doing the thinking for them;
- **rigor** — the student is succeeding: push further (ask to justify, generalise, try a harder case).

The release has 520 moments (260 + 260) from 112 sessions. Each moment also carries a frozen **student persona**
and the **real continuation** of the conversation, used by the simulated student.

## 2. The replay: 5 generated turns

`tutormoments/src/tutormoments/conversation.py`

| Role | Prompt | Receives |
|---|---|---|
| Tutor (model under test) | `prompts/tutor/plain.txt` **or** `prompts/tutor/scaffolding_rigor.txt` (chosen with `--modes`) | the prompt + the real conversation up to the cut + the turns generated so far |
| Simulated student | `prompts/student/oracle.txt` | the persona + the real continuation as a reference + the turns so far |

The tutor speaks first; 5 turns in total (tutor, student, tutor, student, tutor).

The two tutor prompts:

- **plain** (~70 words): "You are an online tutor… use everything you know about what makes great tutoring."
  No pedagogical guidance.
- **scaffolding_rigor** (~220 words): an "expert tutor" prompt describing the scaffold ↔ rigor dial, and the two
  failure modes the benchmark measures (over-scaffolding, missed chances for rigor).

The student and judge prompts are the same whatever tutor prompt is used.

## 3. The judge: three passes

`tutormoments/src/tutormoments/scoring.py`, prompts in `prompts/scorer/`. Four LLM calls per moment:

| Pass | Prompt | Input | Output |
|---|---|---|---|
| 1. **annotate** | `annotate/scaffolding.md` (~1,600 words: definitions, guidance, 3 examples) | **only the 5 generated turns** (`context_window=0`; earlier turns are replaced by "turns 1–N omitted"), marked as the moment, plus a sentence giving the gold label ("A team of teachers believe that this moment is appropriate for scaffolding") | a coach-style analysis: **Situation** (student's state), **Action** (what the tutor did), **Result** (effect on the student) |
| 2a. **decompose** | `decompose/decompose_action.md` | only the *Action* text of pass 1 | a list of short, atomic tutor actions (hypothetical actions are dropped) |
| 2b. **decompose** | `decompose/decompose_overscaffold.md` | only the *Action* text of pass 1 | the spans that indicate over-scaffolding (empty list = none) |
| 3. **structure** | `structure/classify_action.md` | only the action list of pass 2a | `{"scaffolding": yes/no, "rigor": yes/no}` → label **scaffolding / rigor / both / neither** |

Then the **action taxonomy** (`prompts/taxonomy/classify_actions.md`) puts every atomic action of pass 2a into
one of 13 categories (guiding questions, hints, explaining, supplying answers, prompting for justification, …),
50 actions per call.

There is **one verdict per moment**: the judge looks at all the tutor's turns together. A single turn that gives
the answer away is enough to mark the moment as over-scaffolding. A moment can be both *scaffolding* and
*over-scaffolding* (the tutor scaffolds, then gives the answer).

`decompose_result.md` and `classify_student_result.md` analyse the *Result* part; they are not used by S, R, O.

## 4. The metrics

`tutormoments/src/tutormoments/report.py` (`aggregate`), computed in Python, no LLM:

| Metric | Over which moments | A moment counts as a success if |
|---|---|---|
| **S** — appropriate scaffolding | moments whose gold label is scaffolding | label is scaffolding or both **and** no over-scaffolding |
| **R** — appropriate rigor | moments whose gold label is rigor | label is rigor or both **and** no over-scaffolding |
| **O** — avoids over-scaffolding | all moments | no over-scaffolding (O = 1 − over-scaffolding rate) |

All three are *higher is better*. Reference values from the paper (520 moments, judge Claude Opus):
real human tutors S 0.46, R 0.18, O 0.50; every model improves with the `scaffolding_rigor` prompt.
The paper notes that R is probably underestimated for everyone (the judge finds about 55% of the rigor the
teachers marked).

**KL divergence** (`KL S||R / R||S` in the run summary) compares the action mix (taxonomy categories) in
scaffolding moments with the mix in rigor moments: a tutor that adapts to the situation has a high KL; a value
near 0 means it does the same things everywhere. S and R alone do not reward adapting, because *both* counts
for both.

`compare_runs.py` adds 95% confidence intervals by resampling whole sessions (moments of the same session are not
independent).

## 5. What our code changes

Ai2's code is not edited. `scripts/run_benchmark.py` replaces a few functions **in memory** before starting
Ai2's command line (`tutormoments run`):

1. Every name in `configs/models.yaml` is registered as a valid tutor and student.
2. `resolve_tutor` / `resolve_student` return our client for those names, as a *hosted* model, so Ai2's loop
   still builds the official prompts. Backends: an OpenAI-compatible server (EURECOM gateway, vLLM, Ollama) or
   a model loaded with Hugging Face `transformers`.
3. `ModelClient` / `run_batch` send the judge's requests to the gateway as normal requests, 4 in parallel
   (`JUDGE_CONCURRENCY`), because the gateway has no batch API.
4. Small adaptations, method unchanged:
   - `<think>…</think>` blocks are removed from answers;
   - a markdown fence around the judge's JSON (```` ```json ````) is removed before Ai2's parser; raw judge
     answers are kept in `logs/judge_raw.jsonl`;
   - GLM's hidden reasoning counts against `max_tokens`, so GLM calls always get the full token budget;
   - reasoning level as in the paper's config: the paper runs the student and the taxonomy with thinking
     disabled and the judge with adaptive thinking. On the EURECOM gateway only `reasoning_effort: low` turns
     GLM's reasoning down (`enable_thinking=false` makes it reason in the visible text instead; see
     `scripts/test_thinking.py`). So `glm-student` and the taxonomy use `reasoning_effort: low`, the judge and
     the GLM tutor use the default.

## 6. How to change models and prompts

| To change… | File | What to edit |
|---|---|---|
| add a tutor | `configs/models.yaml` | an entry under `models:`, then `--tutors <name>` |
| the student | `configs/socrate_config.yaml` | `student: { model: <name from configs/models.yaml>, mode: oracle }` |
| the judge | `configs/socrate_config.yaml` | `scorer: { model: <gateway model>, reasoning: none }` |
| the taxonomy model | `configs/socrate_config.yaml` | `taxonomy: { model: …, reasoning: none, batch_size: 50 }` |
| reasoning options | `configs/models.yaml` | `extra_body: {reasoning_effort: low}` on a model, or in the `judge:` block |
| a prompt | `tutormoments/src/tutormoments/prompts/…` | edit on a branch of the submodule and report it: results are no longer comparable with the paper |

Example, a model on a laptop through Ollama:

```yaml
# configs/models.yaml, under models:
  gemma3-4b-ollama:
    endpoint: ollama
    model: gemma3:4b
    max_tokens: 1024
```

Example, a model from the Hugging Face hub on a GPU through vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002`):

```yaml
# configs/models.yaml: add the endpoint under endpoints: and the model under models:
endpoints:
  vllm_extra: {base_url: http://localhost:8002/v1, api_key: EMPTY}
models:
  qwen2.5-7b:
    endpoint: vllm_extra
    model: Qwen/Qwen2.5-7B-Instruct
    max_tokens: 2048
```

A run folder is named `<tutor>_<prompt>_<subset>_<date>`: after changing a prompt or a setting, rename or move
the old folder, otherwise the run resumes it and skips the moments already done.

## 7. Cost and limitations

- Gateway tokens per moment (measured): GLM tutor ~30k, student ~30k, judge ~8.5k, taxonomy ~0.5k.
- **Self-preference**: GLM is judge, student and one of the tutors; its score as a tutor may be inflated. Planned
  check: re-judge a sample with a different judge (`rejudge_benchmark.py` measures agreement with the paper's
  judge on Ai2's published conversations).
- Absolute values are not comparable with the paper's Table 8 (different judge); comparisons between tutors under
  the same judge, student and moments are.
- Small subsets give wide intervals: always read `compare_runs.py`'s confidence intervals.
