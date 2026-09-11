#!/usr/bin/env python3
"""
cg-gateway Terminal Client & Python SDK (Smart Session Pool Edition)
Connects to cg-gateway Web API Proxy via localhost or custom URL.
Zero external dependencies (Pure Python Standard Library only).
"""

import sys
import os
import json
import urllib.request
import urllib.error
import argparse
from typing import Generator, Dict, Any, Tuple, Optional, List

DEFAULT_API_URL = os.getenv("CG_API_BASE", "http://127.0.0.1:8560")

# ANSI Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color(text: str, c: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{c}{text}{RESET}"


class CGCLIClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str = "lemon"):
        self.base_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "CGClient/1.0",
            "Authorization": f"Bearer {self.api_key}"
        }

    def check_health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e), "status": "offline"}

    def new_session(self, conv_id: Optional[str] = None) -> Optional[str]:
        try:
            url = f"{self.base_url}/chat/new"
            if conv_id:
                url += f"?conv_id={conv_id}"
            req = urllib.request.Request(url, data=b"{}", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("session_id")
        except Exception as e:
            print(color(f"[Error] Failed to reset session: {e}", RED))
            return None

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-5-6-thinking",
        thinking: Optional[bool] = None,
        new_session: bool = False,
        session_id: Optional[str] = None,
        user: Optional[str] = None
    ) -> Generator[Tuple[Dict[str, Any], Optional[str]], None, None]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "new_session": new_session
        }
        if session_id:
            payload["session_id"] = session_id
        if user:
            payload["user"] = user
        if thinking is not None:
            payload["thinking"] = thinking

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers()
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                chunk_raw = line_str[6:]
                if chunk_raw == "[DONE]":
                    break
                if not chunk_raw.startswith("{"):
                    continue
                try:
                    chunk = json.loads(chunk_raw)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        sess_id = chunk.get("session_id")
                        yield delta, sess_id
                except Exception:
                    continue


def interactive_repl(client: CGCLIClient, default_model: str = "gpt-5-6-thinking", thinking: Optional[bool] = None):
    print(color("\n=======================================================", CYAN))
    print(color("  cg-gateway Interactive Terminal (Smart Session Pool)  ", BOLD + CYAN))
    print(color("=======================================================", CYAN))
    print(f"API Target : {color(client.base_url, GREEN)}")
    print(f"Model      : {color(default_model, YELLOW)}")
    print(f"Thinking   : {color('ON' if thinking is not False else 'OFF', BLUE)}")
    print("Commands   : /new (reset session), /model <slug>, /exit, /help\n")

    # Verify gateway health
    health = client.check_health()
    if health.get("status") == "online":
        pool_stats = health.get("pool", {})
        print(color(f"[*] Gateway Status: ONLINE | Active Pool: {pool_stats.get('active_conversations_count', 0)}/{pool_stats.get('max_pool_size', 10)}", GREEN))
    else:
        print(color(f"[!] Warning: Gateway seems unreachable at {client.base_url} ({health.get('error', 'offline')})", YELLOW))

    history: List[Dict[str, str]] = []
    current_model = default_model
    current_thinking = thinking
    active_session_id: Optional[str] = None

    while True:
        try:
            prompt = input(color("\nYou > ", BOLD + GREEN)).strip()
            if not prompt:
                continue

            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()
                if cmd in ("/exit", "/quit"):
                    print(color("Goodbye!", CYAN))
                    break
                elif cmd in ("/new", "/reset"):
                    active_session_id = client.new_session()
                    history.clear()
                    print(color("[*] Conversation context cleared and new session initialized.", YELLOW))
                    continue
                elif cmd == "/model":
                    if len(parts) > 1:
                        current_model = parts[1]
                        print(color(f"[*] Switched model to: {current_model}", YELLOW))
                    else:
                        print(color(f"Current model: {current_model}", CYAN))
                    continue
                elif cmd == "/help":
                    print("Available commands:")
                    print("  /new or /reset  - Reset conversation and create fresh session")
                    print("  /model <name>   - Switch model (e.g. gpt-5-6, gpt-5-6-thinking, gpt-5-5)")
                    print("  /exit or /quit  - Exit REPL")
                    continue
                else:
                    print(color(f"Unknown command: {cmd}", RED))
                    continue

            history.append({"role": "user", "content": prompt})

            print(color("Assistant > ", BOLD + BLUE), end="", flush=True)

            in_reasoning = False
            full_response = []
            full_reasoning = []

            for delta, sess_id in client.stream_chat(
                messages=history,
                model=current_model,
                thinking=current_thinking,
                session_id=active_session_id
            ):
                if sess_id:
                    active_session_id = sess_id

                # Handle reasoning tokens
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if not in_reasoning:
                        print(color("\n[Thinking] ", DIM + YELLOW), end="", flush=True)
                        in_reasoning = True
                    print(color(reasoning, DIM + YELLOW), end="", flush=True)
                    full_reasoning.append(reasoning)

                # Handle answer content
                content = delta.get("content")
                if content:
                    if in_reasoning:
                        print("\n", end="", flush=True)
                        in_reasoning = False
                    print(content, end="", flush=True)
                    full_response.append(content)

            print()
            history.append({"role": "assistant", "content": "".join(full_response)})

        except KeyboardInterrupt:
            print(color("\nInterrupted. Type /exit to quit.", YELLOW))
        except Exception as e:
            print(color(f"\n[Error] {e}", RED))


def main():
    parser = argparse.ArgumentParser(description="cg-gateway Terminal Client (Smart Session Pool)")
    parser.add_argument("prompt", nargs="?", help="Direct prompt string. If omitted, opens interactive REPL mode.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"Gateway Base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", default="lemon", help="Gateway access token or PROXY_API_KEY")
    parser.add_argument("--model", default="gpt-5-6-thinking", help="Target model slug (default: gpt-5-6-thinking)")
    parser.add_argument("--no-think", action="store_true", help="Disable extended reasoning (Instant mode)")
    parser.add_argument("--new-session", action="store_true", help="Force fresh session in pool")

    args = parser.parse_args()
    client = CGCLIClient(api_url=args.api_url, api_key=args.api_key)
    thinking = False if args.no_think else None

    if not args.prompt:
        interactive_repl(client, default_model=args.model, thinking=thinking)
        return

    # Single-shot mode
    messages = [{"role": "user", "content": args.prompt}]
    try:
        in_reasoning = False
        for delta, _ in client.stream_chat(
            messages=messages,
            model=args.model,
            thinking=thinking,
            new_session=args.new_session
        ):
            reasoning = delta.get("reasoning_content")
            if reasoning:
                if not in_reasoning:
                    sys.stderr.write(color("[Thinking] ", DIM + YELLOW))
                    in_reasoning = True
                sys.stderr.write(color(reasoning, DIM + YELLOW))
                sys.stderr.flush()

            content = delta.get("content")
            if content:
                if in_reasoning:
                    sys.stderr.write("\n")
                    in_reasoning = False
                sys.stdout.write(content)
                sys.stdout.flush()

        if in_reasoning:
            sys.stderr.write("\n")
        print()
    except Exception as e:
        sys.stderr.write(color(f"[Error] {e}\n", RED))
        sys.exit(1)


if __name__ == "__main__":
    main()
