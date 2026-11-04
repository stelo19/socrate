"""Run the TutorMoments replay with local Ollama models as tutor and student.

Why this script exists
  The benchmark's own "custom tutor" hook (register_tutor) only passes the newly
  generated turns to the tutor, not the real transcript before the cut point, so a
  custom tutor would not know what problem is being discussed. Instead, this script
  plugs Ollama in where the benchmark normally calls a hosted model (OpenAI,
  Anthropic, ...). The tutor and the simulated student therefore receive exactly
  the same prompts as in the paper:
    - tutor:   prompts/tutor/<mode>.txt + transcript up to the cut + generated turns
    - student: prompts/student/oracle.txt + frozen persona + real continuation
  Only the model behind the calls changes.

What it does NOT do
  Scoring (the LLM judge) still uses the paper's setup: Claude via the Anthropic
  Batch API. Without an Anthropic key the run stops at scoring, but the generated
  conversations are already saved in results/<run_id>/transcripts/.

Requirements
  - Ollama running on Windows or in WSL, reachable at http://localhost:11434
  - a pulled model, e.g. `ollama pull llama3.2:3b`
  - a config file whose student block names "ollama-student" (see usage)

Usage (from ~/socrate, venv active):
    sed 's/^student: .*/student: { model: ollama-student, mode: oracle }/' \
        tutormoments/src/tutormoments/default_config.yaml > ollama_config.yaml
    python scripts/run_ollama.py run --config ollama_config.yaml \
        --tutors ollama-tutor --modes plain scaffolding_rigor --sample 2 --concurrency 1

Environment variables (optional)
    OLLAMA_BASE_URL       default http://localhost:11434/v1
    OLLAMA_TUTOR_MODEL    default llama3.2:3b
    OLLAMA_STUDENT_MODEL  default = same as the tutor model
"""

import os
import time
from types import SimpleNamespace

from openai import OpenAI

import tutormoments.conversation as conversation
from tutormoments import register_student, register_tutor
from tutormoments.cli import main
from tutormoments.config import student_spec

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TUTOR_MODEL = os.getenv("OLLAMA_TUTOR_MODEL", "llama3.2:3b")
STUDENT_MODEL = os.getenv("OLLAMA_STUDENT_MODEL", TUTOR_MODEL)

TUTOR_NAME = "ollama-tutor"
STUDENT_NAME = "ollama-student"


class OllamaClient:
    """Minimal stand-in for the benchmark's model client.

    The conversation loop calls client.generate(prompt, cacheable_prefix=..., ...)
    and reads .text, .usage, .latency_seconds and .timing from the result.
    Like the benchmark's OpenAI path, the prompt is sent as ONE user message:
    cacheable_prefix (system prompt + transcript up to the cut) followed by
    prompt (generated turns + "Respond as the TUTOR/STUDENT").
    """

    def __init__(self, model):
        self.model = model
        self._client = OpenAI(api_key="ollama", base_url=BASE_URL)  # Ollama ignores the key

    def generate(self, prompt, *, cacheable_prefix=None, max_tokens=0, **_ignored):
        content = (cacheable_prefix or "") + prompt
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        start = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start

        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
        return SimpleNamespace(text=resp.choices[0].message.content or "",
                               usage=usage, latency_seconds=latency, timing=None)


# Register the two names so the command line accepts them. These functions are
# never called: the patches below route both names to OllamaClient instead.
@register_tutor(TUTOR_NAME)
def _tutor_placeholder(turns):
    raise RuntimeError("routed to Ollama, should not be called")


@register_student(STUDENT_NAME)
def _student_placeholder(turns):
    raise RuntimeError("routed to Ollama, should not be called")


# Patch the two functions the conversation loop uses to pick its models.
# Returning kind="hosted" makes the loop build the official prompts and
# pass the frozen student persona, exactly as for the paper's models.
_original_resolve_tutor = conversation.resolve_tutor
_original_resolve_student = conversation.resolve_student


def resolve_tutor(tutor_id):
    if tutor_id == TUTOR_NAME:
        return {"kind": "hosted", "client": OllamaClient(TUTOR_MODEL), "kwargs": {}}
    return _original_resolve_tutor(tutor_id)


def resolve_student(student_id=None):
    if (student_id or student_spec().model) == STUDENT_NAME:
        return {"kind": "hosted", "client": OllamaClient(STUDENT_MODEL), "kwargs": {}}
    return _original_resolve_student(student_id)


conversation.resolve_tutor = resolve_tutor
conversation.resolve_student = resolve_student

if __name__ == "__main__":
    print(f"Ollama at {BASE_URL} | tutor: {TUTOR_MODEL} | student: {STUDENT_MODEL}")
    main()
