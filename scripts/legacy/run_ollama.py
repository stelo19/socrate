"""Run the TutorMoments benchmark with local / gateway models instead of the paper's APIs.

Why this script exists
  The benchmark's own "custom tutor" hook (register_tutor) only passes the newly
  generated turns to the tutor, not the real transcript before the cut point, so a
  custom tutor would not know what problem is being discussed. Instead, this script
  plugs OpenAI-compatible servers (Ollama, the EURECOM gateway) in where the
  benchmark normally calls a hosted model. The tutor and the simulated student
  therefore receive exactly the same prompts as in the paper:
    - tutor:   prompts/tutor/<mode>.txt + transcript up to the cut + generated turns
    - student: prompts/student/oracle.txt + frozen persona + real continuation
  Only the model behind the calls changes.

Judge (scorer) and action taxonomy
  The paper's judge is Claude Opus 4.6 via the Anthropic Batch API. Gateways and
  Ollama have no batch API, so when GATEWAY_BASE_URL is set, every model the
  benchmark does not recognise (e.g. glm-5.3-flash, or any name listed in
  GATEWAY_MODELS) is called one request at a time on the gateway instead -- the same
  idea as the benchmark's own run_sync_entries(). Prompts and parsing are
  unchanged; only the model behind the calls changes.

Requirements
  - Ollama running and reachable at http://localhost:11434 (for tutor/student)
  - for the judge: a .env with GATEWAY_API_KEY and GATEWAY_BASE_URL
  - a config file whose student block names "ollama-student" and whose scorer /
    taxonomy blocks name the gateway model (see usage)

Usage (from ~/socrate, venv active):
    python scripts/run_ollama.py run --config judge_config.yaml \
        --tutors ollama-tutor --modes plain --sample 1 --concurrency 1

Environment variables (optional unless stated)
    OLLAMA_BASE_URL       default http://localhost:11434/v1
    OLLAMA_TUTOR_MODEL    default llama3.2:3b
    OLLAMA_STUDENT_MODEL  default = same as the tutor model
    GATEWAY_BASE_URL      e.g. https://llm.tools.eurecom.fr/v1 (falls back to EURECOM_BASE_URL)
    GATEWAY_API_KEY       key for the gateway (falls back to EURECOM_API_KEY)
    GATEWAY_MODELS        comma-separated model names to route to the gateway (default:
                          every model name the benchmark does not recognise, e.g. glm-5.3-flash)
    GATEWAY_MAX_TOKENS    cap on output tokens per judge call (default 16384)
    JUDGE_RAW_LOG         file where the judge's raw answers are appended (default judge_raw.jsonl)
"""

import logging
import os
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

import tutormoments.client as client_module
import tutormoments.config as config_module
import tutormoments.conversation as conversation
import tutormoments.models as models_module
from tutormoments import register_student, register_tutor
from tutormoments.cli import main
from tutormoments.config import student_spec
from tutormoments.usage import EMPTY_USAGE

load_dotenv()
logger = logging.getLogger("run_ollama")

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TUTOR_MODEL = os.getenv("OLLAMA_TUTOR_MODEL", "llama3.2:3b")
STUDENT_MODEL = os.getenv("OLLAMA_STUDENT_MODEL", TUTOR_MODEL)

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL") or os.getenv("EURECOM_BASE_URL")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY") or os.getenv("EURECOM_API_KEY")
GATEWAY_MAX_TOKENS = int(os.getenv("GATEWAY_MAX_TOKENS", "16384"))

TUTOR_NAME = "ollama-tutor"
STUDENT_NAME = "ollama-student"


def make_usage(prompt_tokens, completion_tokens, model):
    """Usage dict in the benchmark's format (same keys as tutormoments.usage.EMPTY_USAGE)."""
    usage = dict(EMPTY_USAGE)
    total = prompt_tokens + completion_tokens
    usage.update({"input_tokens": prompt_tokens, "output_tokens": completion_tokens,
                  "total_tokens": total, "input_uncached": prompt_tokens,
                  "output": completion_tokens, "total": total,
                  "provider": "openai-compatible", "model": model, "endpoint": "sync"})
    return usage


class OpenAICompatibleClient:
    """Minimal stand-in for the benchmark's model client, for any OpenAI-compatible server.

    The benchmark calls client.generate(prompt, cacheable_prefix=..., ...) and reads
    .text, .usage, .latency_seconds and .timing from the result. Like the benchmark's
    OpenAI path, the prompt is sent as ONE user message: cacheable_prefix (fixed
    part) followed by prompt (part that changes at every turn).
    """

    def __init__(self, model, base_url, api_key, max_tokens_cap=None):
        self.model = model
        self.provider = "openai-compatible"
        self.max_tokens_cap = max_tokens_cap
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, *, cacheable_prefix=None, max_tokens=0, json_mode=False, **_ignored):
        content = (cacheable_prefix or "") + prompt
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if self.max_tokens_cap:
            max_tokens = min(max_tokens or self.max_tokens_cap, self.max_tokens_cap)
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        start = time.perf_counter()
        if json_mode:
            try:
                resp = self._client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs)
            except Exception:  # server does not support JSON mode: ask for plain text
                resp = self._client.chat.completions.create(**kwargs)
        else:
            resp = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start

        u = resp.usage
        usage = make_usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, self.model)
        return SimpleNamespace(text=resp.choices[0].message.content or "",
                               usage=usage, latency_seconds=latency, timing=None)


def ollama_client(model):
    return OpenAICompatibleClient(model, BASE_URL, "ollama")  # Ollama ignores the key


# ---------------------------------------------------------------- tutor & student
# Register the two names so the command line accepts them. These functions are
# never called: the patches below route both names to Ollama instead.
@register_tutor(TUTOR_NAME)
def _tutor_placeholder(turns):
    raise RuntimeError("routed to Ollama, should not be called")


@register_student(STUDENT_NAME)
def _student_placeholder(turns):
    raise RuntimeError("routed to Ollama, should not be called")


# Returning kind="hosted" makes the conversation loop build the official prompts
# and pass the frozen student persona, exactly as for the paper's models.
_original_resolve_tutor = conversation.resolve_tutor
_original_resolve_student = conversation.resolve_student


def resolve_tutor(tutor_id):
    if tutor_id == TUTOR_NAME:
        return {"kind": "hosted", "client": ollama_client(TUTOR_MODEL), "kwargs": {}}
    return _original_resolve_tutor(tutor_id)


def resolve_student(student_id=None):
    if (student_id or student_spec().model) == STUDENT_NAME:
        return {"kind": "hosted", "client": ollama_client(STUDENT_MODEL), "kwargs": {}}
    return _original_resolve_student(student_id)


conversation.resolve_tutor = resolve_tutor
conversation.resolve_student = resolve_student


# ---------------------------------------------------------------- judge & taxonomy on the gateway
_original_infer_provider = models_module.infer_provider


def is_gateway_model(model):
    """True for models to route to the gateway.

    Either listed in GATEWAY_MODELS, or a name the benchmark cannot map to one of
    its providers (Anthropic, OpenAI, Gemini, Together), e.g. "glm-5.3-flash".
    """
    if not GATEWAY_BASE_URL:
        return False
    listed = {n.strip() for n in os.getenv("GATEWAY_MODELS", "").split(",") if n.strip()}
    if model in listed:
        return True
    try:
        _original_infer_provider(model)
        return False
    except ValueError:
        return True


def infer_provider(model):
    """Replacement for infer_provider(): gateway models count as OpenAI-compatible.

    Needed because the config check refuses model names without a known prefix.
    """
    if is_gateway_model(model):
        return "openai"
    return _original_infer_provider(model)


_OriginalModelClient = client_module.ModelClient
_original_run_batch = client_module.run_batch


def model_client(model):
    """Replacement for ModelClient(model): gateway models get our client, others the original."""
    if is_gateway_model(model):
        return OpenAICompatibleClient(model, GATEWAY_BASE_URL, GATEWAY_API_KEY, GATEWAY_MAX_TOKENS)
    return _OriginalModelClient(model)


def strip_code_fence(text):
    """Remove a markdown code fence (```json ... ```) around a JSON answer.

    The benchmark's parsers expect bare JSON; some models wrap it in a fence.
    Only the wrapper is removed, the content is left untouched.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


RAW_LOG = os.getenv("JUDGE_RAW_LOG", "judge_raw.jsonl")


def log_raw(display_name, key, text, cleaned):
    """Append the judge's raw answer to a local file, to check what the model wrote."""
    import json
    with open(RAW_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pass": display_name, "key": key, "raw": text,
                             "fence_removed": text.strip() != cleaned}, ensure_ascii=False) + "\n")


def run_batch(client, entries, json_mode=True, **kwargs):
    """Replacement for run_batch(): for gateway clients, run the entries one at a time.

    Returns the same shape as the original: {key: {"text": ..., "usage": ...}},
    plus "error" for failed entries, so the scorer's parsing works unchanged.
    """
    if not isinstance(client, OpenAICompatibleClient):
        return _original_run_batch(client, entries, json_mode=json_mode, **kwargs)
    results = {}
    for i, entry in enumerate(entries, 1):
        key, prompt, entry_json, max_tokens, _images = client_module._extract_entry(entry)
        prefix = entry.get("cacheable_prefix") or ""
        logger.info("judge call %d/%d (%s)", i, len(entries), kwargs.get("display_name", "batch"))
        try:
            resp = client.generate(prompt, cacheable_prefix=prefix, max_tokens=max_tokens,
                                   json_mode=entry_json and json_mode)
            cleaned = strip_code_fence(resp.text)
            log_raw(kwargs.get("display_name", "batch"), key, resp.text, cleaned)
            results[key] = {"text": cleaned, "usage": resp.usage}
        except Exception as exc:  # noqa: BLE001 -- keep going, the scorer handles failed entries
            logger.error("judge call failed for %s: %s", key, exc)
            results[key] = {"text": "", "error": str(exc), "usage": dict(EMPTY_USAGE)}
    return results


if GATEWAY_BASE_URL:
    if not GATEWAY_API_KEY:
        raise SystemExit("GATEWAY_BASE_URL is set but GATEWAY_API_KEY is missing (.env)")
    models_module.infer_provider = infer_provider
    config_module.infer_provider = infer_provider
    client_module.infer_provider = infer_provider
    client_module.ModelClient = model_client
    client_module.run_batch = run_batch


if __name__ == "__main__":
    print(f"tutor/student: Ollama at {BASE_URL} ({TUTOR_MODEL} / {STUDENT_MODEL})")
    if GATEWAY_BASE_URL:
        print(f"judge/taxonomy: models without a known provider prefix go to the gateway at {GATEWAY_BASE_URL}")
    else:
        print("judge/taxonomy: paper setup (Anthropic) -- set GATEWAY_BASE_URL to use the gateway")
    main()
