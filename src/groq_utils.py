"""
Shared robust helper for calling Groq via the OpenAI-compatible client, used by
classify.py, agent.py, and eval/judge.py. Handles two real issues seen in practice:

1. Per-minute rate limits (429 errors) are transient -- retry with backoff instead of
   giving up on the row immediately.
2. Models don't always return perfectly clean JSON: sometimes there's trailing text after
   the JSON object ("Extra data" errors), sometimes a raw control character inside a string
   value ("Invalid control character" errors). Both are handled: strict parsing is tried
   first, then a regex extraction of the first balanced {...} block with lenient parsing.
"""
import json
import re
import time


def _extract_json_object(raw: str) -> str:
    """Find the first balanced {...} block in the text, in case there's extra content
    before or after it (models sometimes add stray text despite instructions)."""
    start = raw.find("{")
    if start == -1:
        return raw
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return raw[start:]  # unbalanced -- return what we have, will likely fail to parse


def parse_json_response(raw: str) -> dict:
    raw = raw.strip().strip("```json").strip("```").strip()
    if not raw:
        raise ValueError("Empty response from model")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback 1: extract just the {...} block in case of trailing/leading text
    candidate = _extract_json_object(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fallback 2: same, but lenient about raw control characters inside strings
    return json.loads(candidate, strict=False)


def call_json_with_retry(client, model: str, prompt: str, max_tokens: int,
                          extra_body: dict = None, max_retries: int = 3) -> dict:
    """Calls the model and parses a JSON response, retrying with backoff on rate limits
    and on JSON parse failures (a retry often just gets a cleaner response)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body or {},
            )
            raw = resp.choices[0].message.content
            return parse_json_response(raw)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                # try to respect the suggested wait time if present, else a safe default
                wait = 12
                match = re.search(r"try again in ([\d.]+)s", err_str)
                if match:
                    wait = float(match.group(1)) + 1
                time.sleep(wait)
            else:
                time.sleep(1)  # brief pause before retrying a parse failure
    raise last_err
