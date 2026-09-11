#!/usr/bin/env python3
"""
test_chatgpt_live.py - Standalone Verification & Client Tool for cg-gateway.
Tests OpenAI-compatible completions against https://chatgpt.indrayuda.my.id
with GPT-5.6 Sol Thinking High (thinking_effort="extended").

Supports:
  - Real-time SSE Streaming with color-coded Thinking vs Content output
  - Buffered (non-streaming) mode
  - Interactive multi-turn or custom single prompts
  - Both Cloudflare Live domain and Local origin testing
"""

import sys
import os
import json
import time
import argparse
from typing import Optional, Dict, Any
import urllib.request
import urllib.error

# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_MAGENTA = "\033[35m"
COLOR_RED = "\033[31m"
COLOR_BLUE = "\033[34m"

DEFAULT_LIVE_URL = "https://chatgpt.indrayuda.my.id/v1/chat/completions"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8560/v1/chat/completions"
DEFAULT_MODEL = "gpt-5-6-thinking"
DEFAULT_KEY = os.getenv("OPENAI_API_KEY", "lemon")
DEFAULT_PROMPT = "Berapa huruf r dalam kata strawberry? Tunjukkan analisis setiap huruf dan posisinya secara bertahap."


def print_banner(target_url: str, model: str, stream: bool, prompt: str):
    print(f"\n{COLOR_MAGENTA}{COLOR_BOLD}{'='*72}{COLOR_RESET}")
    print(f"{COLOR_MAGENTA}{COLOR_BOLD}  CG-GATEWAY LIVE TEST CLIENT -- GPT-5.6 SOL THINKING HIGH{COLOR_RESET}")
    print(f"{COLOR_MAGENTA}{COLOR_BOLD}{'='*72}{COLOR_RESET}")
    print(f"  {COLOR_BOLD}Endpoint:{COLOR_RESET} {COLOR_CYAN}{target_url}{COLOR_RESET}")
    print(f"  {COLOR_BOLD}Model:{COLOR_RESET}    {COLOR_GREEN}{model}{COLOR_RESET} (thinking_effort=extended)")
    print(f"  {COLOR_BOLD}Mode:{COLOR_RESET}     {COLOR_YELLOW}{'Streaming SSE' if stream else 'Buffered JSON'}{COLOR_RESET}")
    print(f"  {COLOR_BOLD}Prompt:{COLOR_RESET}   {COLOR_RESET}{prompt}{COLOR_RESET}")
    print(f"{COLOR_MAGENTA}{'='*72}{COLOR_RESET}\n")


def run_streaming_test(url: str, api_key: str, model: str, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "test_chatgpt_live/1.0.0"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "thinking": True,
        "reasoning_effort": "high"
    }
    if session_id:
        payload["session_id"] = session_id

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    t0 = time.time()
    first_token_time = None
    reasoning_chunks = []
    content_chunks = []
    captured_session_id = None

    print(f"{COLOR_DIM}[Connecting to upstream gateway...]{COLOR_RESET}\n")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            in_thinking_block = False
            in_content_block = False

            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    if "session_id" in chunk and chunk["session_id"]:
                        captured_session_id = chunk["session_id"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Extract reasoning tokens
                    reasoning_delta = delta.get("reasoning_content")
                    if reasoning_delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        if not in_thinking_block:
                            print(f"{COLOR_YELLOW}{COLOR_BOLD}Thinking / Reasoning Process:{COLOR_RESET}")
                            print(f"{COLOR_YELLOW}{COLOR_DIM}", end="", flush=True)
                            in_thinking_block = True
                        print(reasoning_delta, end="", flush=True)
                        reasoning_chunks.append(reasoning_delta)

                    # Extract text content tokens
                    content_delta = delta.get("content")
                    if content_delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        if in_thinking_block:
                            print(f"{COLOR_RESET}\n")
                            in_thinking_block = False
                        if not in_content_block:
                            print(f"{COLOR_GREEN}{COLOR_BOLD}Assistant Response:{COLOR_RESET}")
                            print(f"{COLOR_CYAN}", end="", flush=True)
                            in_content_block = True
                        print(content_delta, end="", flush=True)
                        content_chunks.append(content_delta)

            if in_thinking_block:
                print(f"{COLOR_RESET}")
            if in_content_block:
                print(f"{COLOR_RESET}")

    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="ignore")
        print(f"\n{COLOR_RED}{COLOR_BOLD}[HTTP Error {he.code}]: {err_body}{COLOR_RESET}\n")
        return {"error": f"HTTP {he.code}: {err_body}"}
    except Exception as ex:
        print(f"\n{COLOR_RED}{COLOR_BOLD}[Network Error]: {ex}{COLOR_RESET}\n")
        return {"error": str(ex)}

    total_time = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time else 0.0

    full_reasoning = "".join(reasoning_chunks)
    full_content = "".join(content_chunks)

    print(f"\n{COLOR_MAGENTA}{'-'*72}{COLOR_RESET}")
    print(f"{COLOR_BOLD}Execution Telemetry:{COLOR_RESET}")
    print(f"  - TTFT (Time To First Token): {COLOR_YELLOW}{ttft:.2f}s{COLOR_RESET}")
    print(f"  - Total Elapsed Time:        {COLOR_YELLOW}{total_time:.2f}s{COLOR_RESET}")
    print(f"  - Reasoning Output Length:   {COLOR_YELLOW}{len(full_reasoning)} chars{COLOR_RESET}")
    print(f"  - Content Output Length:     {COLOR_CYAN}{len(full_content)} chars{COLOR_RESET}")
    if captured_session_id:
        print(f"  - Upstream Session ID:       {COLOR_GREEN}{captured_session_id}{COLOR_RESET}")
    print(f"{COLOR_MAGENTA}{'-'*72}{COLOR_RESET}\n")

    return {
        "status": "success",
        "ttft": ttft,
        "total_time": total_time,
        "reasoning": full_reasoning,
        "content": full_content,
        "session_id": captured_session_id
    }


def run_buffered_test(url: str, api_key: str, model: str, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "test_chatgpt_live/1.0.0"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "thinking": True,
        "reasoning_effort": "high"
    }
    if session_id:
        payload["session_id"] = session_id

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    t0 = time.time()
    print(f"{COLOR_DIM}[Sending buffered request to upstream gateway...]{COLOR_RESET}")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="ignore")
        print(f"\n{COLOR_RED}{COLOR_BOLD}[HTTP Error {he.code}]: {err_body}{COLOR_RESET}\n")
        return {"error": f"HTTP {he.code}: {err_body}"}
    except Exception as ex:
        print(f"\n{COLOR_RED}{COLOR_BOLD}[Network Error]: {ex}{COLOR_RESET}\n")
        return {"error": str(ex)}

    total_time = time.time() - t0
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    reasoning = message.get("reasoning_content")
    content = message.get("content", "")
    sid = data.get("session_id")
    usage = data.get("usage", {})

    if reasoning:
        print(f"\n{COLOR_YELLOW}{COLOR_BOLD}Reasoning Content:{COLOR_RESET}")
        print(f"{COLOR_YELLOW}{COLOR_DIM}{reasoning}{COLOR_RESET}\n")

    print(f"{COLOR_GREEN}{COLOR_BOLD}Assistant Response:{COLOR_RESET}")
    print(f"{COLOR_CYAN}{content}{COLOR_RESET}\n")

    print(f"{COLOR_MAGENTA}{'-'*72}{COLOR_RESET}")
    print(f"{COLOR_BOLD}Buffered Response Telemetry:{COLOR_RESET}")
    print(f"  - Total Elapsed Time:        {COLOR_YELLOW}{total_time:.2f}s{COLOR_RESET}")
    print(f"  - Prompt Tokens:             {usage.get('prompt_tokens', 0)}")
    print(f"  - Completion Tokens:         {usage.get('completion_tokens', 0)}")
    print(f"  - Total Tokens:              {usage.get('total_tokens', 0)}")
    if sid:
        print(f"  - Upstream Session ID:       {COLOR_GREEN}{sid}{COLOR_RESET}")
    print(f"{COLOR_MAGENTA}{'-'*72}{COLOR_RESET}\n")

    return {
        "status": "success",
        "total_time": total_time,
        "reasoning": reasoning,
        "content": content,
        "session_id": sid,
        "usage": usage
    }


def interactive_loop(url: str, api_key: str, model: str, stream: bool):
    print(f"\n{COLOR_GREEN}{COLOR_BOLD}Entering Interactive Multi-Turn Mode{COLOR_RESET}")
    print(f"{COLOR_DIM}Type your message and press Enter. Type 'exit' or 'quit' to end session.{COLOR_RESET}\n")
    session_id = None
    turn = 1

    while True:
        try:
            prompt = input(f"{COLOR_BOLD}[Turn {turn}] You:{COLOR_RESET} ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print(f"\n{COLOR_YELLOW}Exiting interactive session.{COLOR_RESET}\n")
                break

            print()
            if stream:
                res = run_streaming_test(url, api_key, model, prompt, session_id=session_id)
            else:
                res = run_buffered_test(url, api_key, model, prompt, session_id=session_id)

            if res.get("status") == "success" and res.get("session_id"):
                session_id = res.get("session_id")
            turn += 1
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{COLOR_YELLOW}Session interrupted.{COLOR_RESET}\n")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Verification client for cg-gateway with GPT-5.6 Sol Thinking High."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_LIVE_URL,
        help=f"Target completions URL (default: {DEFAULT_LIVE_URL})"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Target local gateway endpoint ({DEFAULT_LOCAL_URL})"
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help="Authorization Bearer key (default: $OPENAI_API_KEY or 'lemon')"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to evaluate"
    )
    parser.add_argument(
        "--buffered",
        action="store_true",
        help="Run buffered non-streaming mode instead of streaming SSE"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both Streaming SSE and Buffered tests sequentially"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive conversational multi-turn session"
    )

    args = parser.parse_args()

    target_url = DEFAULT_LOCAL_URL if args.local else args.url

    if args.interactive:
        print_banner(target_url, args.model, stream=not args.buffered, prompt="<Interactive Multi-Turn>")
        interactive_loop(target_url, args.key, args.model, stream=not args.buffered)
        return

    if args.both:
        # Run 1: Streaming Mode
        print_banner(target_url, args.model, stream=True, prompt=args.prompt)
        res_stream = run_streaming_test(target_url, args.key, args.model, args.prompt)
        if res_stream.get("error"):
            sys.exit(1)

        # Run 2: Buffered Mode
        buffered_prompt = "Sebutkan 3 planet terdekat dengan matahari beserta jarak rata-ratanya."
        print_banner(target_url, args.model, stream=False, prompt=buffered_prompt)
        res_buf = run_buffered_test(target_url, args.key, args.model, buffered_prompt)
        if res_buf.get("error"):
            sys.exit(1)

        print(f"{COLOR_GREEN}{COLOR_BOLD}ALL MODES (STREAMING + BUFFERED) PASSED SUCCESSFULLY!{COLOR_RESET}\n")
    else:
        stream_mode = not args.buffered
        print_banner(target_url, args.model, stream=stream_mode, prompt=args.prompt)
        if stream_mode:
            res = run_streaming_test(target_url, args.key, args.model, args.prompt)
        else:
            res = run_buffered_test(target_url, args.key, args.model, args.prompt)

        if res.get("error"):
            sys.exit(1)
        print(f"{COLOR_GREEN}{COLOR_BOLD}TEST COMPLETED SUCCESSFULLY!{COLOR_RESET}\n")


if __name__ == "__main__":
    main()
