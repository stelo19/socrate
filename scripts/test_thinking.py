# Which request options turn GLM's hidden reasoning off on the gateway?
# Sends the same short classification request with different options and prints,
# for each one: finish reason, reasoning length, tokens used, time, start of the answer.
# usage:  python scripts/test_thinking.py [MODEL]      (default glm-5.3-flash)

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("EURECOM_API_KEY"),
                base_url=os.getenv("EURECOM_BASE_URL", "https://llm.tools.eurecom.fr/v1"))
model = sys.argv[1] if len(sys.argv) > 1 else "glm-5.3-flash"

prompt = """Assign each tutor action one category letter: A = asks a guiding question,
B = asks the student to justify, M = other.
1. The tutor asks "what do you notice about the denominators?"
2. The tutor asks the student to explain why the answer is 4/4.
3. The tutor says the internet is slow.
Answer only with JSON: {"assignments": [{"id": 1, "category": "A"}, ...]}"""

schema = {"type": "object", "properties": {"assignments": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "category": {"type": "string"}},
    "required": ["id", "category"]}}}, "required": ["assignments"]}

variants = {
    "default": {},
    "enable_thinking=false": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "thinking disabled": {"extra_body": {"thinking": {"type": "disabled"}}},
    "reasoning_effort=low": {"reasoning_effort": "low"},
    "json_schema (default)": {"response_format": {"type": "json_schema",
                                                  "json_schema": {"name": "output", "schema": schema}}},
}

for name, options in variants.items():
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=4000, **options)
    except Exception as exc:
        print(f"{name:24} ERROR: {str(exc)[:150]}\n")
        continue
    seconds = time.perf_counter() - start
    choice = resp.choices[0]
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    answer = choice.message.content or ""
    print(f"{name:24} finish={choice.finish_reason}  reasoning_chars={len(reasoning)}  "
          f"tokens={resp.usage.completion_tokens}  time={seconds:.1f}s")
    print(f"{'':24} answer: {answer[:120]!r}\n")
