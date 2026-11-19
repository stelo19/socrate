"""Run the TutorMoments benchmark with open models (Hugging Face) and the EURECOM gateway.

What changes with respect to the paper
  Only WHICH MODEL answers. Prompts, conversation loop (5 turns), judge prompts,
  parsing and metrics are Ai2's code, unchanged:
    - tutor:   prompts/tutor/<mode>.txt + real transcript up to the cut + generated turns
    - student: prompts/student/oracle.txt + frozen persona + real continuation
    - judge:   prompts/scorer/* (3 passes), then report.aggregate -> S, R, O

How models are plugged in
  - Tutors and the student are named in configs/models.yaml. Each name points either to
    an OpenAI-compatible server (EURECOM gateway, vLLM, Ollama) or to a model
    loaded in-process with Hugging Face transformers.
  - The tutor is chosen on the command line (--tutors NAME ...); the student is
    the "student" model of the config file (configs/socrate_config.yaml: glm-student).
  - The benchmark asks two functions which model to use (resolve_tutor and
    resolve_student); this script replaces them, in memory only, so that our
    models are treated like the paper's hosted models (official prompts, persona).
  - The judge (config "scorer", glm-5.3-flash) and the action taxonomy run on the
    gateway. The paper's judge uses provider Batch APIs, which the gateway does
    not offer, so judge requests are sent one at a time (same idea as the
    benchmark's own run_sync_entries).

Small adaptations (documented, not changing the method)
  - Reasoning blocks (<think>...</think>) are removed from model answers.
  - A markdown code fence (```json ... ```) around the judge's JSON is removed
    before Ai2's parser reads it; raw judge answers are kept in logs/judge_raw.jsonl.
  - GLM reasons before answering and that reasoning counts against max_tokens, so
    GLM calls always get the full token budget (otherwise the taxonomy's 4000-token
    limit leaves empty answers).
  - Reasoning level as in the paper's config: the student and the taxonomy run with
    reasoning off in the paper, so here they use reasoning_effort "low" (models.yaml:
    glm-student, judge.taxonomy_extra_body); the judge keeps GLM's default reasoning
    (the paper's judge uses adaptive thinking).
  - The action taxonomy asks for structured output (a JSON schema); it is sent as
    response_format json_schema, with json_object / plain text as fallbacks.

Usage (from the project folder, venv active):
    python scripts/run_benchmark.py run --config configs/socrate_config.yaml \
        --tutors glm qwen3-8b llama-3.1-8b --modes plain scaffolding_rigor \
        --data_path subsets/subset_100 --concurrency 4

Environment (.env): EURECOM_API_KEY, EURECOM_BASE_URL; optional VLLM_QWEN_URL,
VLLM_LLAMA_URL, OLLAMA_BASE_URL, SOCRATE_MODELS (path of models.yaml),
JUDGE_RAW_LOG, GATEWAY_MAX_TOKENS, JUDGE_CONCURRENCY (parallel judge calls, default 4).
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml
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
logger = logging.getLogger("socrate")

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = Path(os.getenv("SOCRATE_MODELS", ROOT / "configs" / "models.yaml"))
JUDGE_RAW_LOG = Path(os.getenv("JUDGE_RAW_LOG", ROOT / "logs" / "judge_raw.jsonl"))
GATEWAY_MAX_TOKENS = int(os.getenv("GATEWAY_MAX_TOKENS", "16384"))
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "4"))


# ---------------------------------------------------------------- models.yaml
def expand_env(value):
    """Replace ${VAR} and ${VAR:-default} with environment values."""
    if not isinstance(value, str):
        return value

    def sub(m):
        name, default = m.group(1), m.group(3)
        return os.getenv(name) or (default if default is not None else "")

    return re.sub(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}", sub, value)


def load_registry(path=MODELS_FILE):
    data = yaml.safe_load(open(path, encoding="utf-8"))
    endpoints = {k: {kk: expand_env(vv) for kk, vv in v.items()}
                 for k, v in (data.get("endpoints") or {}).items()}
    return endpoints, data.get("models") or {}


ENDPOINTS, MODELS = load_registry()
GATEWAY = ENDPOINTS.get("gateway", {})
# options for the judge / taxonomy calls (the judge MODEL is set in socrate_config.yaml)
JUDGE_OPTIONS = {"max_tokens": GATEWAY_MAX_TOKENS, "reasoning": True, "extra_body": None,
                 "taxonomy_extra_body": None,
                 **(yaml.safe_load(open(MODELS_FILE, encoding="utf-8")).get("judge") or {})}


# ---------------------------------------------------------------- helpers
def make_usage(prompt_tokens, completion_tokens, model, provider):
    """Usage dict in the benchmark's format (same keys as tutormoments.usage.EMPTY_USAGE)."""
    usage = dict(EMPTY_USAGE)
    total = prompt_tokens + completion_tokens
    usage.update({"input_tokens": prompt_tokens, "output_tokens": completion_tokens,
                  "total_tokens": total, "input_uncached": prompt_tokens,
                  "output": completion_tokens, "total": total,
                  "provider": provider, "model": model, "endpoint": "sync"})
    return usage


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text):
    """Remove <think>...</think> blocks (reasoning models); keep only the answer."""
    text = THINK_RE.sub("", text or "")
    if "<think>" in text:            # unfinished reasoning (ran out of tokens)
        text = text.split("<think>")[0]
    return text.strip()


def strip_code_fence(text):
    """Remove a markdown code fence (```json ... ```) around a JSON answer."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# ---------------------------------------------------------------- clients
class OpenAICompatibleClient:
    """Stand-in for the benchmark's model client, for any OpenAI-compatible server.

    The benchmark calls client.generate(prompt, cacheable_prefix=..., ...) and reads
    .text, .usage, .latency_seconds and .timing. Like the benchmark's OpenAI path,
    the prompt is sent as ONE user message: cacheable_prefix (fixed part) + prompt.
    """

    provider = "openai-compatible"

    def __init__(self, model, base_url, api_key, max_tokens_cap=None, extra_body=None,
                 reasoning=False):
        if not base_url:
            raise SystemExit(f"no base_url for model {model}: check models.yaml / .env")
        self.model = model
        self.max_tokens_cap = max_tokens_cap
        self.extra_body = extra_body or None
        self.schema_extra_body = None
        # reasoning models (GLM) think before answering and the hidden reasoning counts
        # against max_tokens: always give them the full cap, or short limits set by the
        # benchmark (e.g. 4000 for the taxonomy) leave an empty answer
        self.reasoning = reasoning
        self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    def _create(self, kwargs, formats):
        """Try the response formats in order; the last one (None) is plain text."""
        for fmt in formats:
            try:
                if fmt is None:
                    return self._client.chat.completions.create(**kwargs)
                return self._client.chat.completions.create(response_format=fmt, **kwargs)
            except Exception as exc:  # format not supported by this server: try the next one
                if fmt is not None:
                    continue
                if "extra_body" not in kwargs:
                    raise
                # the server refuses the extra options (e.g. a thinking switch): drop them
                logger.warning("%s refused extra_body %s (%s); sending without it",
                               self.model, kwargs["extra_body"], str(exc)[:120])
                self.extra_body = self.schema_extra_body = None
                kwargs = {k: v for k, v in kwargs.items() if k != "extra_body"}
                return self._create(kwargs, formats)

    def generate(self, prompt, *, cacheable_prefix=None, max_tokens=0, json_mode=False,
                 output_schema=None, **_ignored):
        content = (cacheable_prefix or "") + prompt
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if self.max_tokens_cap:
            if self.reasoning:
                max_tokens = self.max_tokens_cap
            else:
                max_tokens = min(max_tokens or self.max_tokens_cap, self.max_tokens_cap)
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        # the taxonomy (the only caller with output_schema) may use its own options
        extra = self.schema_extra_body if output_schema and self.schema_extra_body else self.extra_body
        if extra:
            kwargs["extra_body"] = extra

        # the taxonomy classifier passes a JSON schema (structured output)
        formats = [None]
        if output_schema:
            formats = [{"type": "json_schema",
                        "json_schema": {"name": "output", "schema": output_schema}},
                       {"type": "json_object"}, None]
        elif json_mode:
            formats = [{"type": "json_object"}, None]

        start = time.perf_counter()
        resp = self._create(kwargs, formats)
        latency = time.perf_counter() - start

        u = resp.usage
        usage = make_usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0,
                           self.model, self.provider)
        choice = resp.choices[0]
        text = strip_thinking(choice.message.content)
        info = {"finish_reason": choice.finish_reason,
                "reasoning_chars": len(getattr(choice.message, "reasoning_content", None) or ""),
                "completion_tokens": usage["output_tokens"]}
        if output_schema:
            cleaned = strip_code_fence(text)
            log_raw("taxonomy", None, text, cleaned, info)
            text = cleaned
        return SimpleNamespace(text=text, usage=usage, latency_seconds=latency, timing=None,
                               info=info)


class TransformersClient:
    """Same interface, but the model runs inside this process with Hugging Face transformers.

    The model is loaded once and shared; generation is serialised with a lock, so
    use --concurrency 1 with this backend.
    """

    provider = "transformers"
    _cache = {}
    _lock = threading.Lock()

    def __init__(self, model, max_tokens_cap=1024, enable_thinking=None):
        self.model = model
        self.max_tokens_cap = max_tokens_cap
        self.enable_thinking = enable_thinking
        if model not in TransformersClient._cache:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("loading %s with transformers ...", model)
            tok = AutoTokenizer.from_pretrained(model)
            mdl = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None)
            TransformersClient._cache[model] = (tok, mdl)
        self.tokenizer, self.hf_model = TransformersClient._cache[model]

    def generate(self, prompt, *, cacheable_prefix=None, max_tokens=0, **_ignored):
        content = (cacheable_prefix or "") + prompt
        messages = [{"role": "user", "content": content}]
        template_kwargs = {}
        if self.enable_thinking is not None:          # Qwen3: turn reasoning on/off
            template_kwargs["enable_thinking"] = self.enable_thinking
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, **template_kwargs).to(self.hf_model.device)
        n_new = min(max_tokens or self.max_tokens_cap, self.max_tokens_cap)

        start = time.perf_counter()
        with TransformersClient._lock:
            out = self.hf_model.generate(**inputs, max_new_tokens=n_new)
        latency = time.perf_counter() - start

        n_in = inputs["input_ids"].shape[1]
        new_ids = out[0][n_in:]
        text = strip_thinking(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        usage = make_usage(int(n_in), int(len(new_ids)), self.model, self.provider)
        return SimpleNamespace(text=text, usage=usage, latency_seconds=latency, timing=None)


def build_client(name):
    """Client for a model name from models.yaml."""
    spec = MODELS[name]
    cap = spec.get("max_tokens")
    if spec.get("backend") == "transformers":
        return TransformersClient(spec["model"], cap or 1024, spec.get("enable_thinking"))
    ep = ENDPOINTS[spec["endpoint"]]
    return OpenAICompatibleClient(spec["model"], spec.get("base_url") or ep.get("base_url"),
                                  ep.get("api_key"), cap, spec.get("extra_body"),
                                  spec.get("reasoning", False))


# ---------------------------------------------------------------- tutor & student
# Register every name of models.yaml so the command line accepts it. These
# functions are never called: resolve_tutor/resolve_student below route the
# names to our clients instead.
def _placeholder(turns):
    raise RuntimeError("routed by run_benchmark.py, should not be called")


for _name in MODELS:
    register_tutor(_name)(_placeholder)
    register_student(_name)(_placeholder)

_original_resolve_tutor = conversation.resolve_tutor
_original_resolve_student = conversation.resolve_student


def resolve_tutor(tutor_id):
    # kind="hosted": the loop builds the official prompts, as for the paper's models
    if tutor_id in MODELS:
        return {"kind": "hosted", "client": build_client(tutor_id), "kwargs": {}}
    return _original_resolve_tutor(tutor_id)


def resolve_student(student_id=None):
    name = student_id or student_spec().model
    if name in MODELS:
        return {"kind": "hosted", "client": build_client(name), "kwargs": {}}
    return _original_resolve_student(student_id)


conversation.resolve_tutor = resolve_tutor
conversation.resolve_student = resolve_student


# ---------------------------------------------------------------- judge & taxonomy on the gateway
_original_infer_provider = models_module.infer_provider


def is_gateway_model(model):
    """Models the benchmark cannot map to Anthropic/OpenAI/Gemini/Together (e.g. glm-5.3-flash)."""
    try:
        _original_infer_provider(model)
        return False
    except ValueError:
        return True


def infer_provider(model):
    """Gateway models count as OpenAI-compatible, so the config check accepts them."""
    if is_gateway_model(model):
        return "openai"
    return _original_infer_provider(model)


_OriginalModelClient = client_module.ModelClient
_original_run_batch = client_module.run_batch


def model_client(model):
    """Replacement for ModelClient(model): judge/taxonomy models go to the gateway."""
    if is_gateway_model(model):
        client = OpenAICompatibleClient(model, GATEWAY.get("base_url"), GATEWAY.get("api_key"),
                                        JUDGE_OPTIONS["max_tokens"], JUDGE_OPTIONS["extra_body"],
                                        JUDGE_OPTIONS["reasoning"])
        client.schema_extra_body = JUDGE_OPTIONS["taxonomy_extra_body"]
        return client
    return _OriginalModelClient(model)


_log_lock = threading.Lock()


def log_raw(display_name, key, text, cleaned, info=None):
    JUDGE_RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(JUDGE_RAW_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pass": display_name, "key": key, "raw": text,
                             "fence_removed": (text or "").strip() != cleaned, **(info or {})},
                            ensure_ascii=False) + "\n")


def run_batch(client, entries, json_mode=True, **kwargs):
    """Replacement for run_batch(): for our clients, send the judge entries as normal
    requests, JUDGE_CONCURRENCY at a time (the gateway has no batch API).

    Returns the same shape as the original: {key: {"text": ..., "usage": ...}}
    (plus "error" for failed entries), so Ai2's parsing works unchanged.
    """
    if not isinstance(client, OpenAICompatibleClient):
        return _original_run_batch(client, entries, json_mode=json_mode, **kwargs)
    name = kwargs.get("display_name", "batch")
    done = []

    def judge_one(entry):
        key, prompt, entry_json, max_tokens, _images = client_module._extract_entry(entry)
        prefix = entry.get("cacheable_prefix") or ""
        try:
            resp = client.generate(prompt, cacheable_prefix=prefix, max_tokens=max_tokens,
                                   json_mode=entry_json and json_mode)
            cleaned = strip_code_fence(resp.text)
            log_raw(name, key, resp.text, cleaned, resp.info)
            result = {"text": cleaned, "usage": resp.usage}
        except Exception as exc:  # noqa: BLE001 -- the scorer handles failed entries
            logger.error("judge call failed for %s: %s", key, exc)
            result = {"text": "", "error": str(exc), "usage": dict(EMPTY_USAGE)}
        done.append(key)
        logger.info("judge call %d/%d done (%s)", len(done), len(entries), name)
        return key, result

    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
        return dict(pool.map(judge_one, entries))


models_module.infer_provider = infer_provider
config_module.infer_provider = infer_provider
client_module.infer_provider = infer_provider
client_module.ModelClient = model_client
client_module.run_batch = run_batch


if __name__ == "__main__":
    print(f"models file: {MODELS_FILE} ({', '.join(MODELS)})")
    print(f"gateway: {GATEWAY.get('base_url') or 'NOT SET'} | judge raw log: {JUDGE_RAW_LOG}")
    main()
