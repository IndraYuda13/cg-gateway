#!/usr/bin/env python3
"""
ChatGPT Terminal Client & Verification Tool (Smart Session Pool Edition)
Connects to ChatGPT Web API Gateway via Cloudflare Tunnel or local host.
Supports GPT-5.6 Sol Thinking High (thinking_effort="extended"), Multi-Turn continuity,
Universal Multi-File & Multi-Image Upload (PDF, TXT, CSV, DOCX, PNG, JPG, etc.),
and ChatGPT Incognito / Temporary Chat Mode (default: ON, zero history spam).
Zero external dependencies (Python Standard Library only).
"""

import sys
import os
import json
import time
import base64
import urllib.request
import urllib.error
import urllib.parse
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


def color(text: str, c: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{c}{text}{RESET}"


def guess_mime_type(filename_or_ext: str) -> str:
    """
    Infers MIME type from filename or extension.
    """
    ext = os.path.splitext(filename_or_ext)[1].lower() if "." in filename_or_ext else filename_or_ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".json": "application/json",
        ".xml": "application/xml",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".py": "text/x-python",
        ".js": "application/javascript",
        ".ts": "application/typescript",
        ".css": "text/css",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".doc": "application/msword",
        ".xls": "application/vnd.ms-excel",
        ".ppt": "application/vnd.ms-powerpoint",
        ".zip": "application/zip",
        ".tar": "application/x-tar",
        ".gz": "application/gzip",
    }
    return mime_map.get(ext, "application/octet-stream")


def encode_attachment_source(path_or_url: str, is_image: Optional[bool] = None) -> Tuple[str, str, str, str]:
    """
    Normalizes local file path or web URL into (uri_or_url, filename, mime, kind).
    kind is 'image' or 'file'.
    """
    target = path_or_url.strip()
    if target.startswith("data:"):
        mime = "application/octet-stream"
        fname = f"attachment_{int(time.time())}"
        if ";" in target:
            header = target.split(";", 1)[0].replace("data:", "").strip()
            if "/" in header:
                mime = header
        if is_image is not None:
            kind = "image" if is_image else "file"
        else:
            kind = "image" if mime.startswith("image/") else "file"
        return target, fname, mime, kind

    if target.startswith("http://") or target.startswith("https://"):
        parsed = urllib.parse.urlparse(target)
        fname = os.path.basename(parsed.path) or f"attachment_{int(time.time())}"
        mime = guess_mime_type(fname)
        if is_image is not None:
            kind = "image" if is_image else "file"
        else:
            kind = "image" if mime.startswith("image/") else "file"
        return target, fname, mime, kind

    file_path = target[7:] if target.startswith("file://") else target
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Attachment file not found: {file_path}")

    fname = os.path.basename(file_path)
    mime = guess_mime_type(fname)
    if is_image is not None:
        kind = "image" if is_image else "file"
    else:
        kind = "image" if mime.startswith("image/") else "file"

    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    data_uri = f"data:{mime};name={fname};base64,{b64}"
    return data_uri, fname, mime, kind


def encode_image_source(path_or_url: str) -> str:
    uri, _, _, _ = encode_attachment_source(path_or_url, is_image=True)
    return uri


def encode_file_source(path_or_url: str) -> str:
    uri, _, _, _ = encode_attachment_source(path_or_url, is_image=False)
    return uri


def build_user_content(
    prompt_text: str,
    images: Optional[List[str]] = None,
    files: Optional[List[str]] = None
) -> Union[str, List[Dict[str, Any]]]:
    """
    Builds OpenAI Chat Completions user message content supporting multi-images and multi-files.
    """
    images = [img for img in (images or []) if img]
    files = [f for f in (files or []) if f]

    if not images and not files:
        return prompt_text

    parts: List[Dict[str, Any]] = []
    if prompt_text:
        parts.append({"type": "text", "text": prompt_text})

    for img in images:
        uri, fname, mime, _ = encode_attachment_source(img, is_image=True)
        parts.append({
            "type": "image_url",
            "name": fname,
            "image_url": {"url": uri}
        })

    for f in files:
        uri, fname, mime, _ = encode_attachment_source(f, is_image=False)
        parts.append({
            "type": "file_url",
            "name": fname,
            "mime_type": mime,
            "file_url": {"url": uri, "name": fname, "mime_type": mime}
        })

    if not prompt_text:
        if images and files:
            default_p = "Analisis dan jelaskan file dan gambar ini secara detail."
        elif files:
            default_p = "Analisis dan ringkas dokumen ini secara detail."
        else:
            default_p = "Deskripsikan dan analisis gambar ini secara detail."
        parts.insert(0, {"type": "text", "text": default_p})

    return parts


class ChatGPTCLIClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, api_key: str = DEFAULT_KEY):
        self.base_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "User-Agent": "ChatGPTCLIClient/1.3",
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
        new_session: bool = False,
        incognito: bool = True
    ) -> Generator[Tuple[Dict[str, Any], Optional[str]], None, None]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "new_session": new_session,
            "history_and_training_disabled": incognito
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
        user: Optional[str] = None,
        incognito: bool = True
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "history_and_training_disabled": incognito
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


def run_streaming_verification(
    client: ChatGPTCLIClient,
    model: str,
    prompt: Union[str, List[Dict[str, Any]]],
    incognito: bool = True
) -> bool:
    print(f"\n{color('--- [Running Streaming SSE Verification] ---', MAGENTA + BOLD)}")
    t0 = time.time()
    in_think = False
    in_response = False
    full_reasoning = []
    full_content = []
    captured_sid = None

    try:
        msgs = [{"role": "user", "content": prompt}]
        for delta, sid in client.stream_chat(msgs, model=model, thinking=True, incognito=incognito):
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


def run_buffered_verification(
    client: ChatGPTCLIClient,
    model: str,
    prompt: Union[str, List[Dict[str, Any]]],
    incognito: bool = True
) -> bool:
    print(f"\n{color('--- [Running Buffered JSON Verification] ---', MAGENTA + BOLD)}")
    t0 = time.time()
    try:
        msgs = [{"role": "user", "content": prompt}]
        res = client.buffered_chat(msgs, model=model, thinking=True, incognito=incognito)
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
    force_new: bool = False,
    initial_incognito: bool = True
):
    health = client.check_health()
    is_online = health.get("status") == "online"
    status_str = color("Online ✓", GREEN) if is_online else color(f"Warning ({health.get('error', 'offline')})", RED)
    user_name = health.get("account", {}).get("name", "User")
    active_pool_count = health.get("pool", {}).get("active_conversations_count", 0)
    available_models = health.get("models", [])

    print(color("=" * 68, CYAN))
    print(color("       ChatGPT Terminal Client (Multi-Files & Incognito Mode)       ", BOLD + CYAN))
    print(color("=" * 68, CYAN))
    print(f"  {color('API Endpoint  :', BOLD)} {client.base_url}")
    print(f"  {color('Status        :', BOLD)} {status_str} ({user_name})")
    print(f"  {color('Active Pool   :', BOLD)} {active_pool_count} active conversation slots")
    print(f"  {color('Commands      :', BOLD)} /file <path>, /image <path>, /files, /incognito [on|off], /think [on|off], /new, /exit")
    print(color("-" * 68, CYAN))

    current_model = initial_model
    thinking_on = initial_thinking
    incognito_on = initial_incognito
    current_session_id = None
    force_next_new = force_new
    pending_attachments: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    while True:
        try:
            att_badge = f" | {color(f'Files: {len(pending_attachments)}', MAGENTA)}" if pending_attachments else ""
            incog_badge = color("Incognito: ON", CYAN) if incognito_on else color("Incognito: OFF", DIM)
            status_tag = f"[{color('Model: ' + current_model, GREEN)} | {color('Think: ' + ('ON' if thinking_on else 'OFF'), YELLOW)} | {incog_badge}{att_badge}]"
            prompt = input(f"\n{color('You', BOLD + CYAN)} {status_tag} > ").strip()

            if not prompt:
                continue

            lowered = prompt.lower()

            if lowered in ('/exit', '/quit', 'exit', 'quit'):
                print(color("Sampai jumpa bre! 👋", CYAN))
                break

            elif lowered in ('/new', '/reset'):
                history.clear()
                current_session_id = None
                pending_attachments.clear()
                force_next_new = True
                client.new_session()
                print(color("[✓] Fresh conversation thread initialized (history and attachments cleared).", GREEN))
                continue

            elif lowered == '/clear':
                history.clear()
                current_session_id = None
                pending_attachments.clear()
                force_next_new = True
                client.new_session()
                print(color("[✓] Conversation history and attachment queue cleared.", GREEN))
                continue

            elif lowered in ('/clear files', '/clear file', '/clear image', '/clear attach', '/clear attachments'):
                pending_attachments.clear()
                print(color("[✓] Queued attachments cleared.", GREEN))
                continue

            elif lowered in ('/files', '/attachments'):
                if not pending_attachments:
                    print(color("[i] Attachment queue is empty. Use /file <path> or /image <path> to attach.", YELLOW))
                else:
                    print(color(f"\n[Queued Attachments] ({len(pending_attachments)} files):", MAGENTA + BOLD))
                    for i, att in enumerate(pending_attachments, 1):
                        print(f"  {i}. [{att['type']}] {att['name']} ({att['mime']})")
                continue

            elif lowered.startswith('/image'):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1:
                    imgs = [a for a in pending_attachments if a["type"] == "image"]
                    if imgs:
                        print(color(f"[i] Currently queued images ({len(imgs)}):", CYAN))
                        for a in imgs:
                            print(f"    - {a['name']}")
                    else:
                        print(color("[i] No images queued. Usage: /image <file_path_or_url>", YELLOW))
                    continue
                sub = parts[1].strip()
                if sub.lower() in ("clear", "none", "rm", "delete", "off"):
                    pending_attachments = [a for a in pending_attachments if a["type"] != "image"]
                    print(color("[✓] Queued images cleared.", GREEN))
                    continue
                try:
                    uri, fname, mime, kind = encode_attachment_source(sub, is_image=True)
                    pending_attachments.append({
                        "source": sub,
                        "name": fname,
                        "mime": mime,
                        "type": "image",
                        "uri": uri
                    })
                    print(color(f"[✓] Image attached: {fname}. Queued ({len(pending_attachments)} total files).", GREEN))
                except Exception as ex:
                    print(color(f"[x] Error attaching image: {ex}", RED))
                continue

            elif lowered.startswith(('/file', '/attach')):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1:
                    docs = [a for a in pending_attachments if a["type"] == "file"]
                    if docs:
                        print(color(f"[i] Currently queued documents ({len(docs)}):", CYAN))
                        for a in docs:
                            print(f"    - {a['name']} ({a['mime']})")
                    else:
                        print(color("[i] No documents queued. Usage: /file <path_or_url>", YELLOW))
                    continue
                sub = parts[1].strip()
                if sub.lower() in ("clear", "none", "rm", "delete", "off"):
                    pending_attachments = [a for a in pending_attachments if a["type"] != "file"]
                    print(color("[✓] Queued documents cleared.", GREEN))
                    continue
                try:
                    uri, fname, mime, kind = encode_attachment_source(sub, is_image=False)
                    pending_attachments.append({
                        "source": sub,
                        "name": fname,
                        "mime": mime,
                        "type": kind,
                        "uri": uri
                    })
                    print(color(f"[✓] File attached: {fname} ({mime}). Queued ({len(pending_attachments)} total files).", GREEN))
                except Exception as ex:
                    print(color(f"[x] Error attaching file: {ex}", RED))
                continue

            elif lowered.startswith('/incognito'):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1:
                    print(color(f"[i] ChatGPT Incognito Mode is currently: {'ON' if incognito_on else 'OFF'}", CYAN))
                    continue
                arg = parts[1].strip().lower()
                if arg in ("on", "1", "true", "yes"):
                    incognito_on = True
                    print(color("[✓] ChatGPT Incognito Mode ENABLED (zero history spam, 100% clean account).", GREEN))
                elif arg in ("off", "0", "false", "no"):
                    incognito_on = False
                    print(color("[!] ChatGPT Incognito Mode DISABLED (conversations will be saved in account sidebar).", YELLOW))
                else:
                    print(color("Usage: /incognito [on|off]", YELLOW))
                continue

            elif lowered == '/think on':
                thinking_on = True
                if current_model in ("gpt-5-6", "gpt-5-6-instant"):
                    current_model = "gpt-5-6-thinking"
                elif current_model in ("gpt-5-5", "gpt-5-5-instant"):
                    current_model = "gpt-5-5-thinking"
                print(color("[✓] GPT-5.6 Reasoning mode ENABLED (thinking_effort=extended).", GREEN))
                continue

            elif lowered == '/think off':
                thinking_on = False
                if current_model == "gpt-5-6-thinking":
                    current_model = "gpt-5-6"
                elif current_model == "gpt-5-5-thinking":
                    current_model = "gpt-5-5"
                print(color("[✓] Fast Instant mode ENABLED.", GREEN))
                continue

            elif lowered.startswith('/model'):
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

            elif lowered == '/help':
                print("Commands:")
                print("  /file <path|url>   : Attach document (PDF, TXT, CSV, DOCX, etc.) to next prompt")
                print("  /attach <path|url> : Alias for /file")
                print("  /image <path|url>  : Attach image (PNG, JPG, WEBP, etc.) to next prompt")
                print("  /files             : List all currently queued attachments")
                print("  /clear files       : Clear queued attachments without resetting chat")
                print("  /incognito on|off  : Toggle ChatGPT Incognito / Temporary Chat (default: ON)")
                print("  /think on|off      : Toggle GPT-5.6 reasoning (extended thinking)")
                print("  /model <name>      : Switch model (or '/model' to list models)")
                print("  /new               : Start a fresh conversation thread in pool")
                print("  /exit              : Exit client")
                continue

            # Construct message content (bundle pending attachments if present)
            if pending_attachments:
                img_sources = [a["uri"] for a in pending_attachments if a["type"] == "image"]
                file_sources = [a["uri"] for a in pending_attachments if a["type"] != "image"]
                user_msg_content = build_user_content(prompt, images=img_sources, files=file_sources)
                pending_attachments.clear()
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
                    user=user,
                    incognito=incognito_on
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
        description="ChatGPT Web API Terminal Client (Smart Session Pool & Multi-Files Edition)"
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
        action="append",
        default=[],
        help="Path or URL of image to attach (can be specified multiple times for multi-images)"
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Path or URL of document/file to attach (PDF, TXT, CSV, DOCX, etc. - can be specified multiple times)"
    )
    parser.add_argument(
        "--incognito",
        dest="incognito",
        action="store_true",
        default=True,
        help="Enable ChatGPT Incognito / Temporary Chat mode (default: True, zero chat history spam)"
    )
    parser.add_argument(
        "--no-incognito",
        dest="incognito",
        action="store_false",
        help="Disable ChatGPT Incognito mode (save conversation to account history)"
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

    # 1. Verification mode: --both
    if args.both:
        prompt = args.prompt or args.explicit_prompt or "Berapa huruf r dalam kata strawberry? Tunjukkan analisis setiap huruf dan posisinya secara bertahap."
        print(f"\n{color('================================================================', MAGENTA)}")
        print(f"{color('  CG-GATEWAY DUAL VERIFICATION SUITE -- GPT-5.6 SOL THINKING    ', BOLD + MAGENTA)}")
        print(f"{color('================================================================', MAGENTA)}")
        print(f"  Target URL : {client.base_url}")
        print(f"  Model      : {model}")
        print(f"  Prompt     : {prompt}")
        print(f"  Incognito  : {args.incognito}")
        if args.image:
            print(f"  Images     : {args.image}")
        if args.file:
            print(f"  Files      : {args.file}")
        print("\n")

        test_payload = build_user_content(prompt, images=args.image, files=args.file)
        ok1 = run_streaming_verification(client, model, test_payload, incognito=args.incognito)
        ok2 = run_buffered_verification(client, model, test_payload, incognito=args.incognito)
        if ok1 and ok2:
            print(f"\n{color('ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ✓', GREEN + BOLD)}\n")
            sys.exit(0)
        else:
            print(f"\n{color('VERIFICATION FAILED! ✗', RED + BOLD)}\n")
            sys.exit(1)

    # 2. Non-interactive single shot / piped input
    has_input = bool(args.prompt or args.explicit_prompt or not sys.stdin.isatty())
    has_attachments = bool(args.image or args.file)

    if (has_input or has_attachments) and not args.interactive:
        prompt = args.prompt or args.explicit_prompt
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()

        user_content = build_user_content(prompt or "", images=args.image, files=args.file)

        if args.buffered:
            try:
                res = client.buffered_chat(
                    messages=[{"role": "user", "content": user_content}],
                    model=model,
                    thinking=thinking_on,
                    session_id=args.session_id,
                    user=args.user,
                    incognito=args.incognito
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
                new_session=args.new,
                incognito=args.incognito
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
        force_new=args.new,
        initial_incognito=args.incognito
    )


if __name__ == "__main__":
    main()
