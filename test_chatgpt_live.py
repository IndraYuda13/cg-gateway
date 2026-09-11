#!/usr/bin/env python3
"""
ChatGPT Terminal Client & Verification Tool (Smart Session Pool Edition)
Connects to ChatGPT Web API Gateway via Cloudflare Tunnel or local host.
Supports GPT-5.6 Sol Thinking High (thinking_effort="extended") and multi-turn continuity.
Zero external dependencies (Python Standard Library only).
"""

import sys
import os
import json
import time
import base64
import urllib.request
import urllib.error
import argparse
from typing import Generator, Dict, Any, Tuple, Optional, List, Union

# Enable ANSI escape sequences on Windows console
if sys.platform == "win32":
    os.system("")

DEFAULT_API_URL = os.getenv("CHATGPT_API_BASE", "https://chatgpt.indrayuda.my.id")
DEFAULT_LOCAL_URL = "http://127.0.0.1:8560"
DEFAULT_KEY = os.getenv("OPENAI_API_KEY", "lemon")
DEFAULT_MODEL = "gpt-5-6-thinking"

# ANSI Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def encode_image_source(path_or_url: str) -> str:
    """
    Normalizes local file path or web URL into an OpenAI vision image_url string.
    Local files are converted to base64 data URIs.
    """
    target = path_or_url.strip()
    if target.startswith("data:") or target.startswith("http://") or target.startswith("https://"):
        return target

    file_path = target[7:] if target.startswith("file://") else target
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp"
    }
    mime = mime_map.get(ext, "image/png")

    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def color(text: str, c: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{c}{text}{RESET}"


class ChatGPTCLIClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str = DEFAULT_KEY):
        self.base_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "User-Agent": "ChatGPTCLIClient/1.2",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e), "status": "offline"}

    def get_models(self) -> List[str]:
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

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
        messages: List[Dict[str, Any]],
        model: str = DEFAULT_MODEL,
        thinking: Optional[bool] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        new_session: bool = False
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
            if thinking:
                payload["thinking_effort"] = "extended"
                payload["reasoning_effort"] = "high"

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(accept="text/event-stream")
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str.startswith("data: "):
                    continue
                chunk_raw = line_str[6:].strip()
                if chunk_raw == "[DONE]":
                    break
                if not chunk_raw.startswith("{"):
                    continue
                try:
                    chunk = json.loads(chunk_raw)
                    choices = chunk.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    sess_id = chunk.get("session_id")
                    yield delta, sess_id
                except Exception:
                    continue

    def buffered_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = DEFAULT_MODEL,
        thinking: Optional[bool] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if session_id:
            payload["session_id"] = session_id
        if user:
            payload["user"] = user
        if thinking is not None:
            payload["thinking"] = thinking
            if thinking:
                payload["thinking_effort"] = "extended"
                payload["reasoning_effort"] = "high"

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers()
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))


def run_streaming_verification(client: ChatGPTCLIClient, model: str, prompt: Union[str, List[Dict[str, Any]]]) -> bool:
    print(f"\n{color('--- [Running Streaming SSE Verification] ---', MAGENTA + BOLD)}")
    t0 = time.time()
    in_think = False
    in_response = False
    full_reasoning = []
    full_content = []
    captured_sid = None

    try:
        msgs = [{"role": "user", "content": prompt}]
        for delta, sid in client.stream_chat(msgs, model=model, thinking=True):
            if sid:
                captured_sid = sid
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                if not in_think:
                    sys.stderr.write(color("\n--- [Thinking / GPT-5.6 Reasoning] ---\n", YELLOW + BOLD))
                    in_think = True
                sys.stderr.write(color(r, DIM + YELLOW))
                sys.stderr.flush()
                full_reasoning.append(r)
            c = delta.get("content")
            if c:
                if in_think:
                    sys.stderr.write("\n")
                    in_think = False
                if not in_response:
                    sys.stderr.write(color("\n--- [Response] ---\n", GREEN + BOLD))
                    in_response = True
                sys.stdout.write(c)
                sys.stdout.flush()
                full_content.append(c)
        sys.stdout.write("\n")
        total_time = time.time() - t0
        print(f"{color('[✓] Streaming test passed', GREEN)} in {total_time:.2f}s (Reasoning: {len(''.join(full_reasoning))} chars, Content: {len(''.join(full_content))} chars, Session: {captured_sid})")
        return True
    except Exception as e:
        print(color(f"[x] Streaming test failed: {e}", RED))
        return False


def run_buffered_verification(client: ChatGPTCLIClient, model: str, prompt: Union[str, List[Dict[str, Any]]]) -> bool:
    print(f"\n{color('--- [Running Buffered JSON Verification] ---', MAGENTA + BOLD)}")
    t0 = time.time()
    try:
        msgs = [{"role": "user", "content": prompt}]
        res = client.buffered_chat(msgs, model=model, thinking=True)
        choice = (res.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        r = msg.get("reasoning_content")
        c = msg.get("content", "")
        sid = res.get("session_id")
        total_time = time.time() - t0
        if r:
            print(color("\n--- [Thinking / GPT-5.6 Reasoning] ---", YELLOW + BOLD))
            print(color(r, DIM + YELLOW))
        print(color("\n--- [Response] ---", GREEN + BOLD))
        print(c)
        print(f"\n{color('[✓] Buffered test passed', GREEN)} in {total_time:.2f}s (Session: {sid})")
        return True
    except Exception as e:
        print(color(f"[x] Buffered test failed: {e}", RED))
        return False


def interactive_repl(
    client: ChatGPTCLIClient,
    initial_model: str = DEFAULT_MODEL,
    initial_thinking: bool = True,
    user: Optional[str] = None,
    force_new: bool = False
):
    health = client.check_health()
    is_online = health.get("status") == "online"
    status_str = color("Online ✓", GREEN) if is_online else color(f"Warning ({health.get('error', 'offline')})", RED)
    user_name = health.get("account", {}).get("name", "User")
    active_pool_count = health.get("pool", {}).get("active_conversations_count", 0)
    available_models = health.get("models", [])

    print(color("=" * 64, CYAN))
    print(color("       ChatGPT Terminal Client (Smart Session Pool)        ", BOLD + CYAN))
    print(color("=" * 64, CYAN))
    print(f"  {color('API Endpoint  :', BOLD)} {client.base_url}")
    print(f"  {color('Status        :', BOLD)} {status_str} ({user_name})")
    print(f"  {color('Active Pool   :', BOLD)} {active_pool_count} active conversation slots")
    print(f"  {color('Commands      :', BOLD)} /image <path|url>, /think [on|off], /model [name], /new, /exit")
    print(color("-" * 64, CYAN))

    current_model = initial_model
    thinking_on = initial_thinking
    current_session_id = None
    force_next_new = force_new
    pending_image: Optional[str] = None
    pending_image_name: Optional[str] = None
    history: List[Dict[str, Any]] = []

    while True:
        try:
            img_badge = f" | {color(f'Img: {pending_image_name}', MAGENTA)}" if pending_image and pending_image_name else ""
            status_tag = f"[{color('Model: ' + current_model, GREEN)} | {color('Think: ' + ('ON' if thinking_on else 'OFF'), YELLOW)}{img_badge}]"
            prompt = input(f"\n{color('You', BOLD + CYAN)} {status_tag} > ").strip()

            if not prompt:
                continue

            if prompt.lower() in ('/exit', '/quit', 'exit', 'quit'):
                print(color("Sampai jumpa bre! 👋", CYAN))
                break

            elif prompt.lower() in ('/new', '/clear', '/reset'):
                history.clear()
                current_session_id = None
                pending_image = None
                pending_image_name = None
                force_next_new = True
                client.new_session()
                print(color("[✓] Fresh conversation thread initialized (history cleared).", GREEN))
                continue

            elif prompt.lower().startswith('/image'):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1:
                    if pending_image_name:
                        print(color(f"[i] Currently attached image: {pending_image_name}", CYAN))
                    else:
                        print(color("[i] No image attached. Usage: /image <file_path_or_url>", YELLOW))
                    continue
                sub = parts[1].strip()
                if sub.lower() in ("clear", "none", "rm", "delete", "off"):
                    pending_image = None
                    pending_image_name = None
                    print(color("[✓] Attached image cleared.", GREEN))
                    continue
                try:
                    pending_image = encode_image_source(sub)
                    pending_image_name = sub
                    print(color(f"[✓] Image attached: {sub}. It will be sent with your next prompt.", GREEN))
                except Exception as ex:
                    print(color(f"[x] Error attaching image: {ex}", RED))
                continue

            elif prompt.lower() == '/think on':
                thinking_on = True
                if current_model in ("gpt-5-6", "gpt-5-6-instant"):
                    current_model = "gpt-5-6-thinking"
                elif current_model in ("gpt-5-5", "gpt-5-5-instant"):
                    current_model = "gpt-5-5-thinking"
                print(color("[✓] GPT-5.6 Reasoning mode ENABLED (thinking_effort=extended).", GREEN))
                continue

            elif prompt.lower() == '/think off':
                thinking_on = False
                if current_model == "gpt-5-6-thinking":
                    current_model = "gpt-5-6"
                elif current_model == "gpt-5-5-thinking":
                    current_model = "gpt-5-5"
                print(color("[✓] Fast Instant mode ENABLED.", GREEN))
                continue

            elif prompt.lower().startswith('/model'):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1 or parts[1].strip() in ("list", "ls", ""):
                    models_to_show = available_models or ["gpt-5-6-thinking", "gpt-5-6", "gpt-5-5-thinking", "gpt-5-5", "o3-pro", "gpt-6-pro"]
                    print(color(f"Available models: {', '.join(models_to_show)}", CYAN))
                    print(color(f"Current active model: {current_model}", GREEN))
                    continue
                new_model = parts[1].strip()
                current_model = new_model
                if "thinking" in current_model.lower() or current_model.lower() in ("o3-pro", "gpt-5-6-pro", "gpt-6-pro"):
                    thinking_on = True
                else:
                    thinking_on = False
                print(color(f"[✓] Active model set to: {current_model} (Think: {'ON' if thinking_on else 'OFF'})", GREEN))
                continue

            elif prompt.lower() == '/help':
                print("Commands:")
                print("  /image <path|url> : Attach image to next prompt")
                print("  /image clear      : Clear attached image")
                print("  /think on|off     : Toggle GPT-5.6 reasoning (extended thinking)")
                print("  /model <name>     : Switch model (or '/model' to list models)")
                print("  /new              : Start a fresh chat thread in pool")
                print("  /exit             : Exit client")
                continue

            # Construct message content (multimodal if image attached)
            if pending_image:
                user_msg_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": pending_image}}
                ]
                pending_image = None
                pending_image_name = None
            else:
                user_msg_content = prompt

            history.append({"role": "user", "content": user_msg_content})

            in_think = False
            in_response = False
            ans_buf = []

            try:
                for delta, sid in client.stream_chat(
                    messages=history,
                    model=current_model,
                    thinking=thinking_on,
                    session_id=current_session_id,
                    new_session=force_next_new,
                    user=user
                ):
                    force_next_new = False
                    if sid:
                        current_session_id = sid

                    reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning_chunk:
                        if not in_think:
                            sys.stdout.write(f"\n{color('--- [Thinking / GPT-5.6 Reasoning] ---', YELLOW + BOLD)}\n")
                            in_think = True
                        sys.stdout.write(color(reasoning_chunk, DIM + YELLOW))
                        sys.stdout.flush()

                    content_chunk = delta.get("content")
                    if content_chunk:
                        if in_think:
                            sys.stdout.write(f"\n{color('--- [Response] ---', GREEN + BOLD)}\n")
                            in_think = False
                            in_response = True
                        elif not in_response:
                            sys.stdout.write(f"\n{color('--- [Response] ---', GREEN + BOLD)}\n")
                            in_response = True
                        ans_buf.append(content_chunk)
                        sys.stdout.write(content_chunk)
                        sys.stdout.flush()

                sys.stdout.write("\n")
                if in_think:
                    sys.stdout.write("\n")

                full_reply = "".join(ans_buf)
                if full_reply:
                    history.append({"role": "assistant", "content": full_reply})

            except KeyboardInterrupt:
                print(color("\n[Generation interrupted by user]", YELLOW))
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="ignore")
                print(color(f"\n[HTTP Error {he.code}]: {err_body}", RED))
            except Exception as ex:
                print(color(f"\n[Error] {ex}", RED))

        except KeyboardInterrupt:
            print(color("\n[!] Type /exit to quit.", YELLOW))
        except EOFError:
            print(color("\nSampai jumpa bre! 👋", CYAN))
            break


def main():
    parser = argparse.ArgumentParser(
        description="ChatGPT Web API Terminal Client (Smart Session Pool Edition)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Direct prompt. If omitted, opens interactive REPL mode."
    )
    parser.add_argument(
        "--prompt",
        dest="explicit_prompt",
        default=None,
        help="Explicit prompt string (alternative to positional prompt)"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path or URL of an image to attach to the prompt"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API Base URL (default: {DEFAULT_API_URL})"
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
        "--no-think",
        action="store_true",
        help="Disable GPT-5.6 reasoning (Fast Instant mode)"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force create a new conversation thread in pool"
    )
    parser.add_argument(
        "--user",
        default=None,
        help="Explicit user identifier for session isolation"
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Explicit conversation/session ID to resume"
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
        help="Force interactive mode even if stdin is not a tty"
    )

    args = parser.parse_args()

    api_url = DEFAULT_LOCAL_URL if args.local else args.api_url
    client = ChatGPTCLIClient(api_url=api_url, api_key=args.key)

    thinking_on = not args.no_think
    model = args.model
    if not thinking_on and model == "gpt-5-6-thinking":
        model = "gpt-5-6"

    # Helper to build message content with optional image attachment
    def build_user_content(prompt_text: str) -> Union[str, List[Dict[str, Any]]]:
        if args.image:
            img_uri = encode_image_source(args.image)
            return [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_uri}}
            ]
        return prompt_text

    # 1. Verification mode: --both
    if args.both:
        prompt = args.prompt or args.explicit_prompt or "Berapa huruf r dalam kata strawberry? Tunjukkan analisis setiap huruf dan posisinya secara bertahap."
        print(f"\n{color('================================================================', MAGENTA)}")
        print(f"{color('  CG-GATEWAY DUAL VERIFICATION SUITE -- GPT-5.6 SOL THINKING    ', BOLD + MAGENTA)}")
        print(f"{color('================================================================', MAGENTA)}")
        print(f"  Target URL : {client.base_url}")
        print(f"  Model      : {model}")
        print(f"  Prompt     : {prompt}")
        if args.image:
            print(f"  Image      : {args.image}\n")
        else:
            print("\n")

        test_payload = build_user_content(prompt)
        ok1 = run_streaming_verification(client, model, test_payload)
        ok2 = run_buffered_verification(client, model, test_payload)
        if ok1 and ok2:
            print(f"\n{color('ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ✓', GREEN + BOLD)}\n")
            sys.exit(0)
        else:
            print(f"\n{color('VERIFICATION FAILED! ✗', RED + BOLD)}\n")
            sys.exit(1)

    # 2. Non-interactive single shot / piped input
    if (args.prompt or args.explicit_prompt or not sys.stdin.isatty()) and not args.interactive:
        prompt = args.prompt or args.explicit_prompt
        if not prompt:
            prompt = sys.stdin.read().strip()
        if not prompt and not args.image:
            print(color("Prompt cannot be empty unless an image is provided.", RED))
            sys.exit(1)
        if not prompt and args.image:
            prompt = "Deskripsikan gambar ini secara detail."

        user_content = build_user_content(prompt)

        if args.buffered:
            try:
                res = client.buffered_chat(
                    messages=[{"role": "user", "content": user_content}],
                    model=model,
                    thinking=thinking_on,
                    session_id=args.session_id,
                    user=args.user
                )
                choice = (res.get("choices") or [{}])[0]
                msg = choice.get("message", {})
                r = msg.get("reasoning_content")
                c = msg.get("content", "")
                if r:
                    sys.stderr.write(color("\n--- [Thinking / GPT-5.6 Reasoning] ---\n", YELLOW + BOLD))
                    sys.stderr.write(color(r, DIM + YELLOW))
                    sys.stderr.write("\n")
                if c:
                    sys.stderr.write(color("\n--- [Response] ---\n", GREEN + BOLD))
                    sys.stdout.write(c)
                    sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception as e:
                print(color(f"\n[Error] {e}", RED))
                sys.exit(1)
            return

        # Single-shot streaming mode
        in_think = False
        in_response = False
        try:
            msgs = [{"role": "user", "content": user_content}]
            for delta, _ in client.stream_chat(
                messages=msgs,
                model=model,
                thinking=thinking_on,
                session_id=args.session_id,
                user=args.user,
                new_session=args.new
            ):
                reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_chunk:
                    if not in_think:
                        sys.stderr.write(color("\n--- [Thinking / GPT-5.6 Reasoning] ---\n", YELLOW + BOLD))
                        in_think = True
                    sys.stderr.write(color(reasoning_chunk, DIM + YELLOW))
                    sys.stderr.flush()

                content_chunk = delta.get("content")
                if content_chunk:
                    if in_think:
                        sys.stderr.write("\n")
                        in_think = False
                    if not in_response:
                        sys.stderr.write(color("\n--- [Response] ---\n", GREEN + BOLD))
                        in_response = True
                    sys.stdout.write(content_chunk)
                    sys.stdout.flush()

            sys.stdout.write("\n")
        except Exception as e:
            print(color(f"\n[Error] {e}", RED))
            sys.exit(1)
        return

    # 3. Interactive REPL Mode (default when run with no arguments)
    interactive_repl(
        client=client,
        initial_model=model,
        initial_thinking=thinking_on,
        user=args.user,
        force_new=args.new
    )


if __name__ == "__main__":
    main()
