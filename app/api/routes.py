import os
import time
import uuid
import json
import base64
import socket
import ipaddress
import urllib.request
import urllib.parse
from typing import Optional, Any, Generator, Tuple, List, Dict
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import (
    DEFAULT_TOKEN,
    PROXY_API_KEY,
    ACCOUNT_ID,
    DEFAULT_MODELS
)
from app.core.session import smart_pool
from app.core.client import ChatGPTUpstreamClient, inspect_image, guess_mime_type
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceItem,
    ChatMessage,
    UsageInfo,
    ModelListResponse,
    ModelItem
)

router = APIRouter()


def authenticate(authorization: Optional[str] = None) -> str:
    token = DEFAULT_TOKEN
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        bearer = authorization.split("Bearer ", 1)[1].strip()
        if PROXY_API_KEY and bearer != PROXY_API_KEY and not bearer.startswith("eyJ"):
            raise HTTPException(status_code=401, detail="Invalid Proxy API Key")
        if bearer and bearer not in ("lemon", "default", "sk-123", "none", PROXY_API_KEY):
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
    Extracts user prompt text and resolves any embedded image or document attachments from the latest user message.
    Supports ContentPart types: text, image_url, file_url, file, image.
    """
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

    prompt = " ".join([t.strip() for t in text_parts if t.strip()])
    if not prompt and attachments:
        has_docs = any(not (a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal") for a in attachments)
        has_imgs = any(a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal" for a in attachments)
        if has_docs and has_imgs:
            prompt = "Analisis dan jelaskan file dan gambar ini secara detail."
        elif has_docs:
            prompt = "Analisis dan ringkas dokumen ini secara detail."
        else:
            prompt = "Deskripsikan dan analisis gambar ini secara detail."

    return prompt, attachments


def sse_event_stream(
    req: ChatCompletionRequest,
    token: str,
    conv_id: str,
    session_id: str,
    parent_msg_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    history_and_training_disabled: bool = True
) -> Generator[str, None, None]:
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    client = ChatGPTUpstreamClient(token=token)

    # Initial chunk with assistant role
    active_session_id = req.session_id or session_id
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

    # Only supply conversation_id to upstream if this is turn >= 2 with established parent_message_id
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
                history_and_training_disabled=history_and_training_disabled
            ):
                yield ev
        except Exception as ex:
            err_str = str(ex)
            if ("404" in err_str or "not found" in err_str.lower()) and conv_id_for_upstream is not None:
                # Upstream conversation expired or pruned - reset and retry turn fresh
                smart_pool.reset_conv(conv_id)
                for ev in client.stream_chat(
                    prompt=prompt,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id="client-created-root",
                    conversation_id=None,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=history_and_training_disabled
                ):
                    yield ev
            else:
                raise

    try:
        for event in stream_with_retry():
            e_type = event.get("type")
            current_sid = req.session_id or last_conv_id or session_id

            if e_type == "text":
                content = event.get("content", "")
                if content:
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

        # Update smart pool state with returned upstream conversation and message ids
        if last_conv_id:
            smart_pool.update_session_id(conv_id, last_conv_id)
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
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None)
):
    token = authenticate(authorization)
    client = ChatGPTUpstreamClient(token=token)

    # Extract user prompt and any attached files or images
    last_user_prompt, attachments = extract_prompt_and_attachments(req.messages, client)

    if not last_user_prompt and not attachments:
        raise HTTPException(status_code=400, detail="No user message found in request")

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
                prompt=last_user_prompt,
                attachments=attachments,
                history_and_training_disabled=incognito_mode
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

        try:
            completion_res = client.chat_completion(
                prompt=last_user_prompt,
                model=req.model or "gpt-5-6-thinking",
                parent_message_id=parent_id_to_use,
                conversation_id=conv_id_for_upstream,
                thinking=effective_thinking,
                attachments=attachments,
                history_and_training_disabled=incognito_mode
            )
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "not found" in err_msg.lower():
                # Stale upstream conversation DAG - reset and retry once cleanly
                smart_pool.reset_conv(conv_id)
                completion_res = client.chat_completion(
                    prompt=last_user_prompt,
                    model=req.model or "gpt-5-6-thinking",
                    parent_message_id="client-created-root",
                    conversation_id=None,
                    thinking=effective_thinking,
                    attachments=attachments,
                    history_and_training_disabled=incognito_mode
                )
            else:
                raise

        final_conv_id = completion_res.get("conversation_id")
        final_msg_id = completion_res.get("message_id")

        if final_conv_id:
            smart_pool.update_session_id(conv_id, final_conv_id)
        if final_msg_id:
            smart_pool.update_parent(conv_id, final_msg_id)

        content = completion_res.get("content", "")
        reasoning_content = completion_res.get("reasoning_content")

        prompt_tok_est = max(1, len(last_user_prompt) // 4)
        comp_tok_est = max(1, len(content) // 4)

        resp_session_id = req.session_id or session_id or final_conv_id

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
                        content=content,
                        reasoning_content=reasoning_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tok_est,
                completion_tokens=comp_tok_est,
                total_tokens=prompt_tok_est + comp_tok_est
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")
