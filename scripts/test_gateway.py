# Connection test for the EURECOM LLM gateway
# usage:  python scripts/test_gateway.py           -> list available models
#         python scripts/test_gateway.py MODEL     -> list models + one test call
# needs a .env file with EURECOM_API_KEY=... (and optionally EURECOM_BASE_URL)

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("EURECOM_API_KEY")
base_url = os.getenv("EURECOM_BASE_URL", "https://llm.tools.eurecom.fr/v1")

if api_key is None:
    print("EURECOM_API_KEY missing in .env")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

# available models
models = [m.id for m in client.models.list()]
print("available models:", len(models))
for m in sorted(models):
    print(" ", m)

# test call (only if a model name is given)
if len(sys.argv) > 1:
    model = sys.argv[1]
    messages = [
        {"role": "system", "content": "You are a patient math tutor for a 5th grader."},
        {"role": "user", "content": "I think 3/4 + 1/4 = 4/8. Is that right?"},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=200)

    print("\nanswer from", model)
    print(resp.choices[0].message.content)
    print("\ntokens used:", resp.usage.total_tokens)
