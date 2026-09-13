import os
import time
import uuid
import json
import base64
import socket
import secrets
import ipaddress
import urllib.request
import urllib.parse
import asyncio
from typing import Optional, Any, AsyncGenerator, Tuple, List, Dict, Union, Set
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import (
    DEFAULT_TOKEN,
    PROXY_API_KEY,
    ACCOUNT_ID,
    DEFAULT_MODELS
)
from app.core.session import smart_pool
from app.core.client import ChatGPTUpstreamClient, inspect_image, guess_mime_type
from app.core.tools import (
    generate_delimiters,
    sanitize_user_prompt,
    compile_tool_prompt,
    extract_tool_calls_from_text,
    format_tool_definitions,
    CODEX_BACKEND_EXECUTION_PROMPT
)
from app.core.stream_parser import LookaheadStreamParser
from app.core.mcp_bridge import mcp_bridge, MCPSecurityError
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceItem,
    ChatMessage,
    UsageInfo,
    ModelListResponse,
    ModelItem,
    ToolCall,
    ToolCallFunction,
    ResponsesRequest
)
from app.core.responses_adapter import (
    flatten_and_normalize_tools,
    normalize_input_to_messages,
    extract_custom_tool_input,
    ResponsesStreamAdapter,
    format_sse
)

router = APIRouter()


def openai_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    )


def authenticate(authorization: Optional[str] = None) -> str:
    token = DEFAULT_TOKEN
    if PROXY_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        bearer = authorization.split("Bearer ", 1)[1].strip()
        is_key_match = secrets.compare_digest(bearer, PROXY_API_KEY)
        is_jwt = bearer.startswith("eyJ")
        if not is_key_match and not is_jwt:
            raise HTTPException(status_code=401, detail="Invalid Proxy API Key")
        if is_jwt:
            token = bearer
    elif isinstance(authorization, str) and authorization.startswith("Bearer "):
        bearer = authorization.split("Bearer ", 1)[1].strip()
        if bearer.startswith("eyJ"):
            token = bearer
    return token



@router.get("/health")
@router.get("/")
def health(authorization: Optional[str] = Header(default=None)):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)
    user_info = {}
    try:
        user_info = client.get_user_profile()
    except Exception as e:
        user_info = {"error": str(e)}

    active_convs = [
        {
            "conv_id": k,
            "session_id": v.session_id,
            "parent_message_id": v.parent_message_id,
            "turn_count": v.turn_count,
            "last_active": v.last_active
        }
        for k, v in smart_pool.pool.items()
    ]

    models = client.list_models()
    model_ids = [m["id"] for m in models]

    return {
        "status": "online",
        "service": "cg-gateway",
        "version": "1.0.0",
        "account_id": ACCOUNT_ID,
        "models": model_ids,
        "pool": {
            "active_conversations_count": len(smart_pool.pool),
            "max_pool_size": smart_pool.max_size,
            "conversations": active_convs
        },
        "account": {
            "name": user_info.get("name", "Unknown"),
            "email": user_info.get("email", "Unknown"),
            "plan": "Enterprise / Workspace"
        }
    }


@router.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)
    models = client.list_models()
    return {
        "object": "list",
        "data": models
    }


@router.post("/v1/chat/sessions/new")
@router.post("/chat/new")
def new_session(
    conv_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)

    target_conv = conv_id or f"manual_{int(time.time())}"
    new_sid, _ = smart_pool.acquire(
        conv_id=target_conv,
        create_fn=client.create_session,
        delete_fn=client.delete_conversation,
        force_new=True
    )
    return {
        "status": "ok",
        "message": "New session created in pool successfully.",
        "conv_id": target_conv,
        "session_id": new_sid
    }


@router.get("/v1/mcp/tools")
def list_mcp_tools(authorization: Optional[str] = Header(default=None)):
    authenticate(authorization)
    try:
        tools = mcp_bridge.list_tools()
        return {"object": "list", "data": tools}
    except Exception as e:
        return openai_error_response(500, f"Failed to list MCP tools: {e}", error_type="mcp_error")


@router.post("/v1/mcp/call")
def call_mcp_tool(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None)
):
    authenticate(authorization)
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not name:
        return openai_error_response(400, "Missing tool 'name' in MCP call payload", code="invalid_tool_call")
    try:
        res = mcp_bridge.call_tool(name, arguments)
        return {"status": "ok", "result": res}
    except MCPSecurityError as se:
        return openai_error_response(403, str(se), error_type="security_violation", code="mcp_sandbox_violation")
    except Exception as e:
        return openai_error_response(500, f"MCP execution error: {e}", error_type="mcp_error")


def process_attachment_source(
    src: str,
    client: ChatGPTUpstreamClient,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decodes or downloads file/image bytes from base64 data URI, remote HTTP(S) URL, or local path.
    Enforces SSRF prevention, 10s timeout, and 20MB limit.
    Detects MIME type and dimensions (for images) and uploads to ChatGPT upstream via client.upload_file.
    """
    target = (src or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="Empty attachment source provided")

    data_bytes: bytes = b""
    detected_mime: Optional[str] = mime_type
    target_filename: Optional[str] = file_name

    # Case 1: Base64 data URI (e.g. data:application/pdf;name=doc.pdf;base64,...)
    if target.startswith("data:"):
        try:
            if "," in target:
                header, b64_str = target.split(",", 1)
                # Parse header parameters e.g. data:application/pdf;name=doc.pdf;base64
                if ":" in header:
                    header_content = header.split(":", 1)[1]
                    parts = header_content.split(";")
                    for p in parts:
                        p = p.strip()
                        if p.lower() == "base64":
                            continue
                        elif p.lower().startswith("name="):
                            if not target_filename:
                                target_filename = p.split("=", 1)[1].strip('"\'')
                        elif "/" in p and not detected_mime:
                            detected_mime = p
                data_bytes = base64.b64decode(b64_str)
            else:
                data_bytes = base64.b64decode(target)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 attachment: {str(e)}")

    # Case 2: Remote HTTP/HTTPS URL
    elif target.startswith("http://") or target.startswith("https://"):
        parsed = urllib.parse.urlparse(target)
        host = parsed.hostname
        if not host:
            raise HTTPException(status_code=400, detail="Invalid attachment URL host")

        if not target_filename:
            path_base = os.path.basename(parsed.path)
            if path_base:
                target_filename = urllib.parse.unquote(path_base)

        # SSRF Protection: Reject private/loopback/reserved IPs
        try:
            ip_obj = ipaddress.ip_address(host)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
                raise HTTPException(status_code=400, detail=f"Blocked private/internal IP in URL: {host}")
        except ValueError:
            # Host is domain name, resolve and check IP
            try:
                resolved_ip = socket.gethostbyname(host)
                ip_obj = ipaddress.ip_address(resolved_ip)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
                    raise HTTPException(status_code=400, detail=f"Blocked internal resolution for host: {host} -> {resolved_ip}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to resolve host {host}: {e}")

        # Fetch with 10s timeout and 20MB limit
        try:
            req = urllib.request.Request(
                target,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cg-gateway/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > 20 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Attachment exceeds maximum allowed size of 20MB")
                ct = resp.headers.get("Content-Type")
                if ct and "/" in ct and not detected_mime:
                    detected_mime = ct.split(";")[0].strip()
                data_bytes = resp.read(20 * 1024 * 1024 + 1)
                if len(data_bytes) > 20 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Attachment exceeds maximum allowed size of 20MB")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch attachment from URL: {str(e)}")

    # Case 3: Local file path or file:// URI
    else:
        file_path = target[7:] if target.startswith("file://") else target
        if os.path.exists(file_path):
            if not target_filename:
                target_filename = os.path.basename(file_path)
            try:
                with open(file_path, "rb") as f:
                    data_bytes = f.read(20 * 1024 * 1024 + 1)
                    if len(data_bytes) > 20 * 1024 * 1024:
                        raise HTTPException(status_code=400, detail="Local file exceeds 20MB limit")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read local file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Attachment file not found: {target}")

    if not data_bytes:
        raise HTTPException(status_code=400, detail="Empty attachment data received")

    # Determine MIME type and dimensions
    is_img = False
    width, height = None, None

    # Check image magic bytes or if mime says image
    if (detected_mime and detected_mime.startswith("image/")) or (
        data_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        or data_bytes.startswith(b"\xff\xd8")
        or data_bytes.startswith(b"GIF8")
        or (data_bytes.startswith(b"RIFF") and b"WEBP" in data_bytes[:16])
    ):
        w, h, img_mime = inspect_image(data_bytes)
        if img_mime:
            is_img = True
            width, height = w, h
            detected_mime = img_mime

    if not detected_mime:
        if target_filename:
            detected_mime = guess_mime_type(target_filename)
        else:
            detected_mime = "application/octet-stream"

    # Generate filename if still missing
    if not target_filename:
        ext = detected_mime.split("/")[-1] if "/" in detected_mime else "bin"
        prefix = "image" if is_img else "file"
        target_filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"

    use_case = "multimodal" if is_img else "my_files"

    uploaded = client.upload_file(
        file_bytes=data_bytes,
        file_name=target_filename,
        mime_type=detected_mime,
        width=width,
        height=height,
        use_case=use_case
    )
    return uploaded


def process_image_source(image_src: str, client: ChatGPTUpstreamClient) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for process_attachment_source.
    """
    return process_attachment_source(src=image_src, client=client)


def extract_prompt_and_attachments(
    messages: List[Any],
    client: ChatGPTUpstreamClient
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts user prompt text or trailing role: 'tool' messages into upstream-compatible format,
    and resolves any embedded image or document attachments.
    """
    if not messages:
        return "", []

    # Check for trailing role: "tool" messages (Multi-Turn Function Calling Cycle)
    trailing_tool_msgs = []
    for m in reversed(messages):
        role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
        if role == "tool":
            trailing_tool_msgs.append(m)
        else:
            break

    if trailing_tool_msgs:
        trailing_tool_msgs.reverse()
        # Find assistant tool calls to resolve tool name if not provided on the tool message
        assistant_calls_by_id: Dict[str, str] = {}
        for m in messages:
            role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
            if role == "assistant":
                tc_list = getattr(m, "tool_calls", None) if hasattr(m, "tool_calls") else (m.get("tool_calls") if isinstance(m, dict) else None)
                if tc_list and isinstance(tc_list, list):
                    for tc in tc_list:
                        call_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                        fn_obj = getattr(tc, "function", None) if hasattr(tc, "function") else (tc.get("function") if isinstance(tc, dict) else None)
                        fn_name = getattr(fn_obj, "name", None) if hasattr(fn_obj, "name") else (fn_obj.get("name") if isinstance(fn_obj, dict) else None)
                        if call_id and fn_name:
                            assistant_calls_by_id[call_id] = fn_name

        formatted_tool_parts: List[str] = []
        for tm in trailing_tool_msgs:
            t_call_id = getattr(tm, "tool_call_id", None) if hasattr(tm, "tool_call_id") else (tm.get("tool_call_id") if isinstance(tm, dict) else None)
            t_name = getattr(tm, "name", None) if hasattr(tm, "name") else (tm.get("name") if isinstance(tm, dict) else None)
            if not t_name and t_call_id:
                t_name = assistant_calls_by_id.get(t_call_id, "function")
            t_name = t_name or "function"
            t_call_id = t_call_id or "call_unknown"
            t_content = getattr(tm, "content", "") if hasattr(tm, "content") else (tm.get("content", "") if isinstance(tm, dict) else "")
            if not isinstance(t_content, str):
                t_content = json.dumps(t_content)
            formatted_tool_parts.append(f"[Tool Result for {t_name} ({t_call_id})]: {t_content}")

        tool_prompt = "\n\n".join(formatted_tool_parts)
        return tool_prompt, []

    # Standard User Message Extraction
    last_user_msg = None
    for m in reversed(messages):
        role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
        if role == "user":
            last_user_msg = m
            break

    if not last_user_msg:
        return "", []

    content = getattr(last_user_msg, "content", None) if hasattr(last_user_msg, "content") else (last_user_msg.get("content") if isinstance(last_user_msg, dict) else None)
    text_parts: List[str] = []
    attachments: List[Dict[str, Any]] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                p_type = part.get("type", "")
                if p_type == "text":
                    t = part.get("text", "")
                    if t:
                        text_parts.append(t)
                elif p_type == "image_url":
                    img_val = part.get("image_url")
                    url_str = ""
                    fname = part.get("name")
                    if isinstance(img_val, dict):
                        url_str = img_val.get("url", "")
                        fname = img_val.get("name") or fname
                    elif isinstance(img_val, str):
                        url_str = img_val
                    if url_str:
                        att = process_attachment_source(url_str, client, file_name=fname)
                        attachments.append(att)
                elif p_type == "file_url":
                    file_val = part.get("file_url")
                    url_str = ""
                    fname = part.get("name")
                    fmime = part.get("mime_type")
                    if isinstance(file_val, dict):
                        url_str = file_val.get("url", "")
                        fname = file_val.get("name") or fname
                        fmime = file_val.get("mime_type") or fmime
                    elif isinstance(file_val, str):
                        url_str = file_val
                    if url_str:
                        att = process_attachment_source(url_str, client, file_name=fname, mime_type=fmime)
                        attachments.append(att)
                elif p_type in ("image", "file"):
                    src_str = part.get("image") or part.get("file") or part.get("url") or part.get("source", "")
                    fname = part.get("name")
                    fmime = part.get("mime_type")
                    if src_str:
                        att = process_attachment_source(src_str, client, file_name=fname, mime_type=fmime)
                        attachments.append(att)
            elif hasattr(part, "type"):
                p_type = getattr(part, "type")
                if p_type == "text":
                    t = getattr(part, "text", "") or ""
                    if t:
                        text_parts.append(t)
                elif p_type == "image_url":
                    img_val = getattr(part, "image_url", None)
                    url_str = ""
                    fname = getattr(part, "name", None)
                    if isinstance(img_val, dict):
                        url_str = img_val.get("url", "")
                        fname = img_val.get("name") or fname
                    elif isinstance(img_val, str):
                        url_str = img_val
                    elif hasattr(img_val, "url"):
                        url_str = getattr(img_val, "url")
                        fname = getattr(img_val, "name", None) or fname
                    if url_str:
                        att = process_attachment_source(url_str, client, file_name=fname)
                        attachments.append(att)
                elif p_type == "file_url":
                    file_val = getattr(part, "file_url", None)
                    url_str = ""
                    fname = getattr(part, "name", None)
                    fmime = getattr(part, "mime_type", None)
                    if isinstance(file_val, dict):
                        url_str = file_val.get("url", "")
                        fname = file_val.get("name") or fname
                        fmime = file_val.get("mime_type") or fmime
                    elif isinstance(file_val, str):
                        url_str = file_val
                    elif hasattr(file_val, "url"):
                        url_str = getattr(file_val, "url")
                        fname = getattr(file_val, "name", None) or fname
                        fmime = getattr(file_val, "mime_type", None) or fmime
                    if url_str:
                        att = process_attachment_source(url_str, client, file_name=fname, mime_type=fmime)
                        attachments.append(att)
                elif p_type in ("image", "file"):
                    src_str = getattr(part, "image", None) or getattr(part, "file", None) or getattr(part, "url", None) or ""
                    fname = getattr(part, "name", None)
                    fmime = getattr(part, "mime_type", None)
                    if src_str:
                        att = process_attachment_source(src_str, client, file_name=fname, mime_type=fmime)
                        attachments.append(att)

    raw_prompt = " ".join([t.strip() for t in text_parts if t.strip()])
    if not raw_prompt and attachments:
        has_docs = any(not (a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal") for a in attachments)
        has_imgs = any(a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal" for a in attachments)
        if has_docs and has_imgs:
            raw_prompt = "Analisis dan jelaskan file dan gambar ini secara detail."
        elif has_docs:
            raw_prompt = "Analisis dan ringkas dokumen ini secara detail."
        else:
            raw_prompt = "Deskripsikan dan analisis gambar ini secara detail."

    prompt = sanitize_user_prompt(raw_prompt)
    return prompt, attachments


async def sse_event_stream(
    req: ChatCompletionRequest,
    token: str,
    conv_id: str,
    session_id: str,
    parent_msg_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    history_and_training_disabled: bool = True,
    start_delimiter: Optional[str] = None,
    end_delimiter: Optional[str] = None,
    web_search: bool = False
) -> AsyncGenerator[str, None]:
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    client = ChatGPTUpstreamClient(token=token)

    active_session_id = req.session_id or session_id
    current_sid = active_session_id
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "session_id": active_session_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    last_conv_id = None
    last_msg_id = None

    conv_id_for_upstream = None
    if parent_msg_id and parent_msg_id != "client-created-root":
        conv_id_for_upstream = session_id

    effective_thinking = req.thinking
    effort = (req.reasoning_effort or req.thinking_effort or "").lower()
    if effective_thinking is None and effort:
        if effort in ("high", "extended", "medium", "max"):
            effective_thinking = True
        elif effort in ("none", "low", "minimal", "off"):
            effective_thinking = False

    parser = LookaheadStreamParser(
        start_delimiter=start_delimiter,
        end_delimiter=end_delimiter
    ) if (start_delimiter and end_delimiter) else None

    def stream_with_retry():
        nonlocal conv_id_for_upstream, parent_msg_id
        try:
            for ev in client.stream_chat(
                prompt=prompt,
                model=req.model or "gpt-5-6-thinking",
                parent_message_id=parent_msg_id or "client-created-root",
                conversation_id=conv_id_for_upstream,
                thinking=effective_thinking,
                attachments=attachments,
                history_and_training_disabled=history_and_training_disabled,
                web_search=web_search
            ):
                yield ev
        except Exception as ex:
            err_str = str(ex)
            if ("404" in err_str or "not found" in err_str.lower()) and conv_id_for_upstream is not None:
                smart_pool.reset_conv(conv_id)
                session_id = ""
                conv_id_for_upstream = None
                for ev in client.stream_chat(
                    prompt=prompt,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id="client-created-root",
                    conversation_id=None,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=history_and_training_disabled,
                    web_search=web_search
                ):
                    yield ev
            else:
                raise

    try:
        async with smart_pool.get_lock(conv_id):
            for event in stream_with_retry():
                e_type = event.get("type")
                current_sid = req.session_id or last_conv_id or session_id

                if e_type == "text":
                    content = event.get("content", "")
                    if content:
                        if parser:
                            parsed_events = parser.feed(content)
                            for pe in parsed_events:
                                if pe["type"] == "text" and pe["content"]:
                                    chunk = {
                                        "id": chat_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": req.model,
                                        "session_id": current_sid,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": pe["content"]},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                elif pe["type"] == "tool_call_delta":
                                    chunk = {
                                        "id": chat_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": req.model,
                                        "session_id": current_sid,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": pe["delta"],
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                        else:
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model,
                                "session_id": current_sid,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                elif e_type == "reasoning":
                    reasoning = event.get("reasoning", "")
                    if reasoning:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "session_id": current_sid,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": reasoning},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                elif e_type == "meta":
                    if event.get("conversation_id"):
                        last_conv_id = event["conversation_id"]
                    if event.get("message_id"):
                        last_msg_id = event["message_id"]

                elif e_type == "done":
                    if event.get("conversation_id"):
                        last_conv_id = event["conversation_id"]
                    if event.get("message_id"):
                        last_msg_id = event["message_id"]

                await asyncio.sleep(0)

            # Update smart pool state with returned upstream conversation and message ids
            if last_conv_id:
                smart_pool.update_session_id(conv_id, last_conv_id, orig_session_id=session_id)
            if last_msg_id:
                smart_pool.update_parent(conv_id, last_msg_id)

    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "not found" in err_msg.lower():
            smart_pool.reset_conv(conv_id)
        err_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "session_id": req.session_id or session_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n[Error from upstream: {err_msg}]"},
                    "finish_reason": "error"
                }
            ]
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"

    # Flush parser and determine terminal finish_reason
    finish_reason = "stop"
    if parser:
        final_events = parser.finish()
        for pe in final_events:
            if pe["type"] == "text" and pe["content"]:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "session_id": current_sid,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": pe["content"]},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            elif pe["type"] == "tool_call_delta":
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "session_id": current_sid,
                    "choices": [
                        {
                            "index": 0,
                            "delta": pe["delta"],
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        finish_reason = parser.finish_reason

    # Terminal completion chunk
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "session_id": req.session_id or last_conv_id or session_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)

    # Validate tools if provided
    has_tools = bool(req.tools and len(req.tools) > 0)
    start_delim: Optional[str] = None
    end_delim: Optional[str] = None
    tool_sys_prompt: str = ""

    if has_tools:
        try:
            _, start_delim, end_delim = generate_delimiters()
            tool_sys_prompt = compile_tool_prompt(
                tools=req.tools,  # type: ignore
                tool_choice=req.tool_choice,
                start_delimiter=start_delim,
                end_delimiter=end_delim,
                parallel_tool_calls=req.parallel_tool_calls
            )
        except Exception as e:
            return openai_error_response(400, f"Invalid tools definition: {e}", code="invalid_tool_schema")

    # Extract user prompt / tool results and any attached files or images
    extracted_prompt, attachments = extract_prompt_and_attachments(req.messages, client)

    if not extracted_prompt and not attachments:
        return openai_error_response(400, "No valid user or tool message found in request")

    # Prepend any system messages from request
    system_parts: List[str] = []
    for m in req.messages:
        role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
        if role == "system":
            c = getattr(m, "content", "") if hasattr(m, "content") else (m.get("content", "") if isinstance(m, dict) else "")
            if isinstance(c, str) and c.strip():
                system_parts.append(c.strip())

    combined_sys = "\n\n".join(system_parts)
    prefix_instructions = []
    if combined_sys:
        prefix_instructions.append(combined_sys)
    if tool_sys_prompt:
        prefix_instructions.append(tool_sys_prompt)

    if prefix_instructions:
        prompt_to_send = "\n\n".join(prefix_instructions) + "\n\n" + extracted_prompt
    else:
        prompt_to_send = extracted_prompt

    incognito_mode = req.history_and_training_disabled if req.history_and_training_disabled is not None else True

    # Compute conversation identity and pool entry
    messages_dicts = [m.model_dump() for m in req.messages]
    conv_id = smart_pool.compute_conv_id(
        messages=messages_dicts,
        user=req.user,
        explicit_session_id=req.session_id
    )

    session_id, parent_msg_id = smart_pool.acquire(
        conv_id=conv_id,
        create_fn=client.create_session,
        delete_fn=client.delete_conversation,
        force_new=bool(req.new_session)
    )

    parent_id_to_use = parent_msg_id or "client-created-root"

    # Streaming mode
    if req.stream:
        return StreamingResponse(
            sse_event_stream(
                req=req,
                token=token,
                conv_id=conv_id,
                session_id=session_id,
                parent_msg_id=parent_id_to_use,
                prompt=prompt_to_send,
                attachments=attachments,
                history_and_training_disabled=incognito_mode,
                start_delimiter=start_delim,
                end_delimiter=end_delim,
                web_search=bool(req.web_search)
            ),
            media_type="text/event-stream"
        )

    # Non-streaming buffered mode
    try:
        conv_id_for_upstream = None
        if parent_id_to_use != "client-created-root":
            conv_id_for_upstream = session_id

        effective_thinking = req.thinking
        effort = (req.reasoning_effort or req.thinking_effort or "").lower()
        if effective_thinking is None and effort:
            if effort in ("high", "extended", "medium", "max"):
                effective_thinking = True
            elif effort in ("none", "low", "minimal", "off"):
                effective_thinking = False

        async with smart_pool.get_lock(conv_id):
            try:
                completion_res = await asyncio.to_thread(
                    client.chat_completion,
                    prompt=prompt_to_send,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id=parent_id_to_use,
                    conversation_id=conv_id_for_upstream,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=incognito_mode,
                    web_search=bool(req.web_search)
                )
            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "not found" in err_msg.lower():
                    smart_pool.reset_conv(conv_id)
                    session_id = None
                    completion_res = await asyncio.to_thread(
                        client.chat_completion,
                        prompt=prompt_to_send,
                        model=req.model or "gpt-5-6-thinking",
                        parent_message_id="client-created-root",
                        conversation_id=None,
                        thinking=effective_thinking,
                        attachments=attachments,
                        history_and_training_disabled=incognito_mode,
                        web_search=bool(req.web_search)
                    )
                else:
                    raise

            final_conv_id = completion_res.get("conversation_id")
            final_msg_id = completion_res.get("message_id")

            if final_conv_id:
                smart_pool.update_session_id(conv_id, final_conv_id, orig_session_id=session_id)
            if final_msg_id:
                smart_pool.update_parent(conv_id, final_msg_id)

        raw_content = completion_res.get("content", "")
        reasoning_content = completion_res.get("reasoning_content")

        tool_calls_objs: Optional[List[ToolCall]] = None
        finish_reason = "stop"
        content_to_return: Optional[str] = raw_content

        if start_delim and end_delim:
            cleaned_text, extracted_calls = extract_tool_calls_from_text(raw_content, start_delim, end_delim)
            if extracted_calls:
                tool_calls_objs = [
                    ToolCall(
                        id=tc["id"],
                        type="function",
                        function=ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in extracted_calls
                ]
                finish_reason = "tool_calls"
                content_to_return = cleaned_text if cleaned_text else None
            else:
                content_to_return = raw_content

        prompt_tok_est = max(1, len(prompt_to_send) // 4)
        comp_tok_est = max(1, len(raw_content) // 4)
        resp_session_id = req.session_id or final_conv_id or session_id

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model or "gpt-5-6-thinking",
            session_id=resp_session_id,
            choices=[
                ChoiceItem(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content_to_return,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls_objs
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tok_est,
                completion_tokens=comp_tok_est,
                total_tokens=prompt_tok_est + comp_tok_est
            )
        )
    except Exception as e:
        return openai_error_response(500, f"Upstream error: {str(e)}", error_type="upstream_error")


async def responses_sse_stream(
    req: ResponsesRequest,
    token: str,
    conv_id: str,
    session_id: str,
    parent_msg_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    history_and_training_disabled: bool = True,
    start_delimiter: Optional[str] = None,
    end_delimiter: Optional[str] = None,
    freeform_tool_names: Optional[Set[str]] = None,
    web_search: bool = False
) -> AsyncGenerator[str, None]:
    created = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    client = ChatGPTUpstreamClient(token=token)

    adapter = ResponsesStreamAdapter(
        response_id=response_id,
        created=created,
        freeform_tool_names=freeform_tool_names
    )

    parser: Optional[LookaheadStreamParser] = None
    if start_delimiter and end_delimiter:
        parser = LookaheadStreamParser(start_delimiter=start_delimiter, end_delimiter=end_delimiter)

    # Initial events: response.created & response.in_progress
    for ev_name, data in adapter.emit_initial():
        yield format_sse(ev_name, data)

    last_conv_id = None
    last_msg_id = None

    conv_id_for_upstream = None
    if parent_msg_id and parent_msg_id != "client-created-root":
        conv_id_for_upstream = session_id

    effective_thinking = req.thinking
    effort = (req.reasoning_effort or req.thinking_effort or "").lower()
    if effective_thinking is None and effort:
        if effort in ("high", "extended", "medium", "max"):
            effective_thinking = True
        elif effort in ("none", "low", "minimal", "off"):
            effective_thinking = False

    def stream_with_retry():
        nonlocal conv_id_for_upstream, parent_msg_id
        try:
            for ev in client.stream_chat(
                prompt=prompt,
                model=req.model or "gpt-5-6-thinking",
                parent_message_id=parent_msg_id or "client-created-root",
                conversation_id=conv_id_for_upstream,
                thinking=effective_thinking,
                attachments=attachments,
                history_and_training_disabled=history_and_training_disabled,
                web_search=web_search
            ):
                yield ev
        except Exception as ex:
            err_str = str(ex)
            if ("404" in err_str or "not found" in err_str.lower()) and conv_id_for_upstream is not None:
                smart_pool.reset_conv(conv_id)
                session_id = ""
                conv_id_for_upstream = None
                for ev in client.stream_chat(
                    prompt=prompt,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id="client-created-root",
                    conversation_id=None,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=history_and_training_disabled,
                    web_search=web_search
                ):
                    yield ev
            else:
                raise

    try:
        async with smart_pool.get_lock(conv_id):
            for event in stream_with_retry():
                e_type = event.get("type")

                if e_type == "text":
                    content = event.get("content", "")
                    if content:
                        if parser:
                            parsed_events = parser.feed(content)
                            for pe in parsed_events:
                                if pe["type"] == "text" and pe["content"]:
                                    for ev, data in adapter.handle_text_delta(pe["content"]):
                                        yield format_sse(ev, data)
                                elif pe["type"] == "tool_call_delta":
                                    for tc_delta in pe["delta"].get("tool_calls", []):
                                        for ev, data in adapter.handle_tool_call_delta(tc_delta):
                                            yield format_sse(ev, data)
                                elif pe["type"] == "tool_call_completed":
                                    for ev, data in adapter.finalize_tool_call(pe["index"], pe["tool_call"]):
                                        yield format_sse(ev, data)
                        else:
                            for ev, data in adapter.handle_text_delta(content):
                                yield format_sse(ev, data)

                elif e_type == "reasoning":
                    reasoning = event.get("reasoning", "")
                    if reasoning:
                        for ev, data in adapter.handle_reasoning_delta(reasoning):
                            yield format_sse(ev, data)

                elif e_type == "meta":
                    if event.get("conversation_id"):
                        last_conv_id = event["conversation_id"]
                    if event.get("message_id"):
                        last_msg_id = event["message_id"]

                elif e_type == "done":
                    if event.get("conversation_id"):
                        last_conv_id = event["conversation_id"]
                    if event.get("message_id"):
                        last_msg_id = event["message_id"]

                await asyncio.sleep(0)

            if parser:
                final_parsed = parser.finish()
                for pe in final_parsed:
                    if pe["type"] == "text" and pe["content"]:
                        for ev, data in adapter.handle_text_delta(pe["content"]):
                            yield format_sse(ev, data)
                    elif pe["type"] == "tool_call_delta":
                        for tc_delta in pe["delta"].get("tool_calls", []):
                            for ev, data in adapter.handle_tool_call_delta(tc_delta):
                                yield format_sse(ev, data)
                    elif pe["type"] == "tool_call_completed":
                        for ev, data in adapter.finalize_tool_call(pe["index"], pe["tool_call"]):
                            yield format_sse(ev, data)

                for idx, completed_tc in enumerate(parser.tool_calls):
                    for ev, data in adapter.finalize_tool_call(idx, completed_tc):
                        yield format_sse(ev, data)

            for ev, data in adapter.finalize_all_and_complete():
                yield format_sse(ev, data)

            if last_conv_id:
                smart_pool.update_session_id(conv_id, last_conv_id, orig_session_id=session_id)
            if last_msg_id:
                smart_pool.update_parent(conv_id, last_msg_id)

    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "not found" in err_msg.lower():
            smart_pool.reset_conv(conv_id)
        for ev, data in adapter.emit_failed(err_msg):
            yield format_sse(ev, data)


@router.post("/v1/responses")
@router.post("/responses")
async def responses_endpoint(
    req: ResponsesRequest,
    authorization: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)

    # 1. Flatten tools & extract custom freeform tool names (including additional_tools in req.input)
    raw_tools = list(req.tools or [])
    if req.input:
        for item in req.input:
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                in_tools = item.get("tools")
                if isinstance(in_tools, list):
                    raw_tools.extend(in_tools)

    normalized_tools, freeform_tool_names = flatten_and_normalize_tools(raw_tools)

    # 2. Normalize input items and optional instructions into MessageItem objects
    messages, input_custom_tools = normalize_input_to_messages(
        input_items=req.input,
        instructions=req.instructions
    )
    freeform_tool_names.update(input_custom_tools)

    # 3. Setup tool calling instructions if tools present
    has_tools = bool(normalized_tools and len(normalized_tools) > 0)
    start_delim: Optional[str] = None
    end_delim: Optional[str] = None
    tool_sys_prompt: str = ""

    if has_tools:
        try:
            _, start_delim, end_delim = generate_delimiters()
            tool_sys_prompt = compile_tool_prompt(
                tools=normalized_tools,  # type: ignore
                tool_choice=req.tool_choice,
                start_delimiter=start_delim,
                end_delimiter=end_delim,
                parallel_tool_calls=req.parallel_tool_calls,
                backend_contract=CODEX_BACKEND_EXECUTION_PROMPT
            )
        except Exception as e:
            return openai_error_response(400, f"Invalid tools definition: {e}", code="invalid_tool_schema")

    # 4. Extract user prompt / tool results and any attached files or images
    extracted_prompt, attachments = extract_prompt_and_attachments(messages, client)

    if not extracted_prompt and not attachments:
        return openai_error_response(400, "No valid user or tool message found in request")

    # 5. Prepend system messages and tool prompt
    system_parts: List[str] = []
    for m in messages:
        if m.role == "system":
            c = m.content
            if isinstance(c, str) and c.strip():
                system_parts.append(c.strip())

    combined_sys = "\n\n".join(system_parts)
    prefix_instructions = []
    if combined_sys:
        prefix_instructions.append(combined_sys)
    if tool_sys_prompt:
        prefix_instructions.append(tool_sys_prompt)

    if prefix_instructions:
        prompt_to_send = "\n\n".join(prefix_instructions) + "\n\n" + extracted_prompt
    else:
        prompt_to_send = extracted_prompt

    incognito_mode = req.history_and_training_disabled if req.history_and_training_disabled is not None else True

    # 6. Session pool acquisition
    messages_dicts = [m.model_dump() for m in messages]
    conv_id = smart_pool.compute_conv_id(
        messages=messages_dicts,
        user=req.user,
        explicit_session_id=req.session_id
    )

    session_id, parent_msg_id = smart_pool.acquire(
        conv_id=conv_id,
        create_fn=client.create_session,
        delete_fn=client.delete_conversation,
        force_new=bool(req.new_session)
    )

    parent_id_to_use = parent_msg_id or "client-created-root"

    # 7. Streaming mode (default True for Responses API)
    if req.stream is not False:
        return StreamingResponse(
            responses_sse_stream(
                req=req,
                token=token,
                conv_id=conv_id,
                session_id=session_id or "",
                parent_msg_id=parent_id_to_use,
                prompt=prompt_to_send,
                attachments=attachments,
                history_and_training_disabled=incognito_mode,
                start_delimiter=start_delim,
                end_delimiter=end_delim,
                freeform_tool_names=freeform_tool_names,
                web_search=bool(req.web_search)
            ),
            media_type="text/event-stream"
        )

    # 8. Non-streaming mode
    try:
        conv_id_for_upstream = None
        if parent_id_to_use != "client-created-root":
            conv_id_for_upstream = session_id

        effective_thinking = req.thinking
        effort = (req.reasoning_effort or req.thinking_effort or "").lower()
        if effective_thinking is None and effort:
            if effort in ("high", "extended", "medium", "max"):
                effective_thinking = True
            elif effort in ("none", "low", "minimal", "off"):
                effective_thinking = False

        async with smart_pool.get_lock(conv_id):
            try:
                completion_res = await asyncio.to_thread(
                    client.chat_completion,
                    prompt=prompt_to_send,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id=parent_id_to_use,
                    conversation_id=conv_id_for_upstream,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=incognito_mode,
                    web_search=bool(req.web_search)
                )
            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "not found" in err_msg.lower():
                    smart_pool.reset_conv(conv_id)
                    session_id = None
                    completion_res = await asyncio.to_thread(
                        client.chat_completion,
                        prompt=prompt_to_send,
                        model=req.model or "gpt-5-6-thinking",
                        parent_message_id="client-created-root",
                        conversation_id=None,
                        thinking=effective_thinking,
                        attachments=attachments,
                        history_and_training_disabled=incognito_mode,
                        web_search=bool(req.web_search)
                    )
                else:
                    raise

        raw_text = completion_res.get("text", "")
        reasoning_content = completion_res.get("reasoning_content")
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        output_items: List[Dict[str, Any]] = []
        out_idx = 0

        if reasoning_content:
            output_items.append({
                "id": f"rs_{response_id}_{out_idx}",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": reasoning_content
                    }
                ]
            })
            out_idx += 1

        extracted_tool_calls: List[Dict[str, Any]] = []
        final_text = raw_text
        if has_tools and start_delim and end_delim:
            final_text, extracted_tool_calls = extract_tool_calls_from_text(
                raw_text, start_delim, end_delim
            )

        if final_text:
            output_items.append({
                "id": f"msg_{response_id}_{out_idx}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": final_text
                    }
                ]
            })
            out_idx += 1

        for tc in extracted_tool_calls:
            t_name = tc.get("function", {}).get("name", "")
            t_args = tc.get("function", {}).get("arguments", "{}")
            t_call_id = tc.get("id", f"call_{uuid.uuid4().hex[:12]}")

            if t_name in freeform_tool_names:
                custom_input = extract_custom_tool_input(t_args)
                output_items.append({
                    "id": f"ctc_{t_call_id}",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": t_call_id,
                    "name": t_name,
                    "input": custom_input
                })
            else:
                output_items.append({
                    "id": f"fc_{t_call_id}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": t_call_id,
                    "name": t_name,
                    "arguments": t_args
                })
            out_idx += 1

        if completion_res.get("conversation_id"):
            smart_pool.update_session_id(conv_id, completion_res["conversation_id"], orig_session_id=session_id)
        if completion_res.get("message_id"):
            smart_pool.update_parent(conv_id, completion_res["message_id"])

        return JSONResponse(
            status_code=200,
            content={
                "id": response_id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "background": False,
                "error": None,
                "output": output_items,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
        )

    except Exception as e:
        return openai_error_response(500, str(e), error_type="server_error")

