import json
import os
import re
import sys
import threading
import time

import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

from llm_cache import load as load_cache, save as save_cache, stage_cache_path

API_URL = os.environ.get(
    "API_URL", "http://10.65.1.119:5004/v1/chat/completions"
)
API_BASE = API_URL.rsplit("/v1/", 1)[0]
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
API_CONNECT_TIMEOUT = int(os.environ.get("API_CONNECT_TIMEOUT", "30"))
API_READ_TIMEOUT = int(os.environ.get("API_READ_TIMEOUT", "900"))
API_MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "3"))
API_WARMUP_TIMEOUT = int(os.environ.get("API_WARMUP_TIMEOUT", "120"))
API_HEALTH_TIMEOUT = int(os.environ.get("API_HEALTH_TIMEOUT", "15"))
API_MAX_TOKENS = int(os.environ.get("API_MAX_TOKENS", "4096"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "15"))
API_ENABLE_THINKING = os.environ.get("API_ENABLE_THINKING", "false").lower() == "true"


def build_api_payload(**kwargs) -> dict:
    payload = dict(kwargs)
    if not API_ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def api_post(payload: dict, timeout: tuple[int, int], label: str) -> requests.Response:
    started = time.monotonic()
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(HEARTBEAT_INTERVAL):
            elapsed = int(time.monotonic() - started)
            print(f"  ... still waiting ({label}, {elapsed}s elapsed)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    finally:
        done.set()


def check_api_health() -> bool:
    print("Checking API availability...")
    try:
        response = requests.get(
            f"{API_BASE}/v1/models",
            timeout=(API_CONNECT_TIMEOUT, API_HEALTH_TIMEOUT),
        )
        response.raise_for_status()
        print("API is reachable.")
        return True
    except RequestException as exc:
        print(f"API health check failed: {exc}", file=sys.stderr)
        print(
            "Server mungkin sedang loading model LLM. "
            "Tunggu hingga model selesai load di server, lalu coba lagi.",
            file=sys.stderr,
        )
        return False


def warmup_model() -> bool:
    print("Warming up model (text-only ping)...")
    payload = build_api_payload(
        model=MODEL,
        temperature=0,
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply OK"}],
    )
    try:
        response = api_post(
            payload,
            timeout=(API_CONNECT_TIMEOUT, API_WARMUP_TIMEOUT),
            label="warmup",
        )
        response.raise_for_status()
        print("Model warmup OK.")
        return True
    except RequestException as exc:
        print(f"Model warmup failed: {exc}", file=sys.stderr)
        return False


def extract_content(data: dict) -> str | None:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None

    for key in ("content", "reasoning"):
        value = message.get(key)
        if value and str(value).strip():
            return str(value).strip()

    return None


def describe_empty_response(data: dict) -> str:
    try:
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "unknown")
        message = choice.get("message", {})
        has_reasoning = bool(message.get("reasoning"))
        has_content = bool(message.get("content"))
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", "?")
        return (
            f"empty response (finish_reason={finish_reason}, "
            f"content={has_content}, reasoning={has_reasoning}, "
            f"completion_tokens={completion_tokens}). "
            "Qwen3.5 mungkin memakai thinking mode — coba API_ENABLE_THINKING=false "
            "atau naikkan API_MAX_TOKENS."
        )
    except (KeyError, IndexError, TypeError):
        return "empty response (unexpected API format)"


def salvage_plates_array(content: str) -> list[dict]:
    """Extract complete plate objects from truncated decompose JSON."""
    match = re.search(r'"plates"\s*:\s*\[', content)
    if not match:
        return []

    plates: list[dict] = []
    i = match.end()
    n = len(content)
    while i < n:
        while i < n and content[i] in " \t\n\r,":
            i += 1
        if i >= n or content[i] != "{":
            break

        depth = 0
        in_str = False
        escape = False
        obj_start = i
        for j in range(i, n):
            ch = content[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        plates.append(json.loads(content[obj_start : j + 1]))
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break
    return plates


def parse_json_response(content: str) -> dict:
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{[\s\S]*\})", content)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    salvaged = salvage_plates_array(content)
    if salvaged:
        return {"plates": salvaged, "_salvaged": True}

    return {"error": "parse_failed", "raw_response": content}


def call_chat(
    messages: list,
    *,
    label: str = "chat",
    max_tokens: int | None = None,
    json_schema: dict | None = None,
) -> tuple[str, dict]:
    """Call the chat endpoint. When ``json_schema`` is given, output is
    constrained to that schema via vLLM structured output (response_format
    json_schema), guaranteeing structurally valid JSON.

    Returns ``(content, meta)`` where meta includes finish_reason."""
    resolved_tokens = max_tokens if max_tokens is not None else API_MAX_TOKENS
    payload = build_api_payload(
        model=MODEL,
        temperature=0,
        max_tokens=resolved_tokens,
        messages=messages,
    )
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "label_spec",
                "schema": json_schema,
                "strict": True,
            },
        }

    cache_path, may_load = stage_cache_path(label)
    if cache_path is not None and may_load:
        cached = load_cache(cache_path)
        if cached is not None:
            print(f"{label} response from cache ({cache_path.name}, {len(cached)} chars).")
            return cached, {"finish_reason": "cache", "max_tokens": resolved_tokens}

    timeout = (API_CONNECT_TIMEOUT, API_READ_TIMEOUT)
    last_error: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print(f"Retrying {label} request ({attempt}/{API_MAX_RETRIES})...")

            response = api_post(payload, timeout=timeout, label=label)
            response.raise_for_status()

            data = response.json()
            content = extract_content(data)
            if content is None:
                raise ValueError(describe_empty_response(data))

            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", "?")
            print(
                f"{label} response received ({len(content)} chars, "
                f"finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
                f"max_tokens={resolved_tokens})."
            )

            if cache_path is not None:
                save_cache(cache_path, label, content)

            return content, {
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "max_tokens": resolved_tokens,
            }
        except ReadTimeout as exc:
            last_error = exc
            print(
                f"Read timeout after {API_READ_TIMEOUT}s "
                f"(attempt {attempt}/{API_MAX_RETRIES})",
                file=sys.stderr,
            )
        except (ConnectionError, ValueError, RequestException) as exc:
            last_error = exc
            print(
                f"Request failed: {exc} (attempt {attempt}/{API_MAX_RETRIES})",
                file=sys.stderr,
            )

    raise last_error if last_error else RuntimeError(f"{label} request failed")
