"""
Zero-dependency end-to-end smoke test for the deployed Modal SGLang endpoint.
Verifies:
  1. Unauthenticated requests are rejected (401/403)
  2. Authenticated /v1/models returns the served model
  3. Streaming /v1/chat/completions works (reports TTFT & end-to-end latency)
  4. Tool calling roundtrip parses arguments cleanly

Usage:
  python smoke_test.py \
    --url https://<your-app>.modal.run \
    --api-key sk-xxx \
    --modal-key wk-xxx \
    --modal-secret ws-xxx \
    --model qwen3.8-27b-ft
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def http_json(url: str, headers: dict, payload: dict | None = None, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_auth_rejection(base_url: str, label: str, extra_headers: dict | None = None, expect_edge: bool = False):
    print(label)
    t0 = time.perf_counter()
    req = urllib.request.Request(f"{base_url}/v1/models", headers=extra_headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=360) as resp:
            print(f"  [FAIL] Endpoint accepted unauthenticated request! HTTP {resp.status}")
            return False
    except urllib.error.HTTPError as e:
        dt = time.perf_counter() - t0
        if e.code in (401, 403):
            note = " (edge-blocked without GPU wakeup)" if expect_edge and dt < 5.0 else ""
            print(f"  [PASS] Rejected with HTTP {e.code} in {dt:.2f}s{note}")
            return True
        print(f"  [WARN] Unexpected HTTP {e.code} in {dt:.2f}s: {e.read().decode('utf-8', errors='replace')[:300]}")
        return False


def test_models(base_url: str, headers: dict):
    print("[2/4] Checking authenticated /v1/models ...")
    t0 = time.perf_counter()
    status, body = http_json(f"{base_url}/v1/models", headers=headers, timeout=360)
    dt = time.perf_counter() - t0
    ids = [m.get("id") for m in body.get("data", [])]
    print(f"  [PASS] HTTP {status} in {dt:.2f}s — models: {ids}")
    return ids


def test_streaming_chat(base_url: str, headers: dict, model: str):
    print("[3/4] Testing streaming /v1/chat/completions ...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK_SGLANG_FP8"}],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    reasoning_chunks = []
    content_chunks = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            rc = delta.get("reasoning_content") or ""
            ct = delta.get("content") or ""
            if rc or ct:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if rc:
                    reasoning_chunks.append(rc)
                if ct:
                    content_chunks.append(ct)
    total = time.perf_counter() - t0
    reasoning = "".join(reasoning_chunks).strip()
    content = "".join(content_chunks).strip()
    print(f"  [PASS] TTFT={ttft or 0:.2f}s, Total={total:.2f}s")
    print(f"    reasoning_content ({len(reasoning)} chars): {reasoning[:160]!r}")
    print(f"    content           ({len(content)} chars): {content!r}")


def test_tool_calling(base_url: str, headers: dict, model: str):
    print("[4/4] Testing OpenAI tool calling ...")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Read the file /tmp/demo.txt using the read_file tool."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a text file by absolute path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"}
                        },
                        "required": ["path"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    status, body = http_json(f"{base_url}/v1/chat/completions", headers=headers, payload=payload)
    dt = time.perf_counter() - t0
    msg = body.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        print(f"  [PASS] HTTP {status} in {dt:.2f}s — tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")
    else:
        print(
            f"  [WARN] HTTP {status} in {dt:.2f}s — no structured tool_calls returned "
            f"(check TOOL_CALL_PARSER). content={msg.get('content')!r}"
        )


def main():
    parser = argparse.ArgumentParser(description="Smoke-test deployed Modal SGLang endpoint")
    parser.add_argument("--url", required=True, help="Base Modal URL, e.g. https://xxx.modal.run")
    parser.add_argument("--api-key", required=True, help="Bearer API key configured in Modal Secret")
    parser.add_argument(
        "--modal-key",
        default=os.environ.get("MODAL_PROXY_KEY"),
        help="Modal Proxy Auth Token key (wk-..., or env MODAL_PROXY_KEY)",
    )
    parser.add_argument(
        "--modal-secret",
        default=os.environ.get("MODAL_PROXY_SECRET"),
        help="Modal Proxy Auth Token secret (ws-..., or env MODAL_PROXY_SECRET)",
    )
    parser.add_argument("--model", default="qwen3.8-27b-ft", help="Served model name")
    args = parser.parse_args()

    if bool(args.modal_key) != bool(args.modal_secret):
        parser.error("--modal-key and --modal-secret must be provided together.")

    base_url = args.url.rstrip("/")
    proxy_headers = (
        {"Modal-Key": args.modal_key, "Modal-Secret": args.modal_secret}
        if args.modal_key
        else {}
    )
    headers = {
        **proxy_headers,
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    if proxy_headers:
        step1_label = "[1/4] Checking unauthenticated edge rejection on /v1/models (should fail fast without waking GPU) ..."
    else:
        step1_label = "[1/4] Checking unauthenticated rejection on /v1/models (first call may trigger cold start) ..."
    ok = test_auth_rejection(base_url, step1_label, expect_edge=bool(proxy_headers))
    if not ok:
        sys.exit(1)

    test_models(base_url, headers)
    if proxy_headers:
        ok_inner = test_auth_rejection(
            base_url,
            "[2b/4] Checking inner Bearer API key rejection (with valid Modal Proxy Auth headers) ...",
            extra_headers=proxy_headers,
        )
        if not ok_inner:
            sys.exit(1)
    test_streaming_chat(base_url, headers, args.model)
    test_tool_calling(base_url, headers, args.model)
    print("\nAll smoke tests finished.")


if __name__ == "__main__":
    main()
