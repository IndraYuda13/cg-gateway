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
from app.core.client import ChatGPTUpstreamClient, inspect_image
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


def process_image_source(image_src: str, client: ChatGPTUpstreamClient) -> Dict[str, Any]:
    """
    Decodes or downloads image bytes from base64 data URI, remote HTTP(S) URL, or local path.
    Enforces SSRF prevention, 10s timeout, and 20MB limit.
    Inspects image dimensions and uploads to ChatGPT upstream via client.upload_file.
    """
    src = (image_src or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="Empty image source provided")

    img_bytes: bytes = b""
    mime_type: str = "image/png"

    # Case 1: Base64 data URI (e.g. data:image/png;base64,...)
    if src.startswith("data:"):
        try:
            if "," in src:
                header, b64_str = src.split(",", 1)
                if ";" in header and ":" in header:
                    mime_type = header.split(":", 1)[1].split(";", 1)[0].strip()
                img_bytes = base64.b64decode(b64_str)
            else:
                img_bytes = base64.b64decode(src)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

    # Case 2: Remote HTTP/HTTPS URL
    elif src.startswith("http://") or src.startswith("https://"):
        parsed = urllib.parse.urlparse(src)
        host = parsed.hostname
        if not host:
            raise HTTPException(status_code=400, detail="Invalid image URL host")

        # SSRF Protection: Reject private/loopback/reserved IPs
        try:
            ip_obj = ipaddress.ip_address(host)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
                raise HTTPException(status_code=400, detail=f"Blocked private/internal IP in image URL: {host}")
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
                raise HTTPException(status_code=400, detail=f"Failed to resolve image host {host}: {e}")

        # Fetch with 10s timeout and 20MB limit
        try:
            req = urllib.request.Request(
                src,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cg-gateway/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > 20 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Image exceeds maximum allowed size of 20MB")
                ct = resp.headers.get("Content-Type")
                if ct and "/" in ct:
                    mime_type = ct.split(";")[0].strip()
                img_bytes = resp.read(20 * 1024 * 1024 + 1)
                if len(img_bytes) > 20 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Image exceeds maximum allowed size of 20MB")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

    # Case 3: Local file path or file:// URI
    else:
        file_path = src[7:] if src.startswith("file://") else src
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    img_bytes = f.read(20 * 1024 * 1024 + 1)
                    if len(img_bytes) > 20 * 1024 * 1024:
                        raise HTTPException(status_code=400, detail="Local image exceeds 20MB limit")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read local image file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported image format or missing file")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image data received")

    width, height, detected_mime = inspect_image(img_bytes)
    final_mime = detected_mime or mime_type
    ext = final_mime.split("/")[-1] if "/" in final_mime else "png"
    file_name = f"image_{uuid.uuid4().hex[:8]}.{ext}"

    uploaded = client.upload_file(
        file_bytes=img_bytes,
        file_name=file_name,
        mime_type=final_mime,
        width=width,
        height=height,
        use_case="multimodal"
    )
    return uploaded


def extract_prompt_and_attachments(
    messages: List[Any],
    client: ChatGPTUpstreamClient
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts user prompt text and resolves any embedded image attachments from the latest user message.
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
                    text_parts.append(part.get("text", ""))
                elif p_type == "image_url":
                    img_val = part.get("image_url")
                    url_str = ""
                    if isinstance(img_val, dict):
                        url_str = img_val.get("url", "")
                    elif isinstance(img_val, str):
                        url_str = img_val
                    if url_str:
                        att = process_image_source(url_str, client)
                        attachments.append(att)
                elif p_type in ("image", "file"):
                    src_str = part.get("image") or part.get("file") or part.get("url", "")
                    if src_str:
                        att = process_image_source(src_str, client)
                        attachments.append(att)
            elif hasattr(part, "type"):
                p_type = getattr(part, "type")
                if p_type == "text":
                    text_parts.append(getattr(part, "text", "") or "")
                elif p_type == "image_url":
                    img_val = getattr(part, "image_url", None)
                    url_str = ""
                    if isinstance(img_val, dict):
                        url_str = img_val.get("url", "")
                    elif isinstance(img_val, str):
                        url_str = img_val
                    elif hasattr(img_val, "url"):
                        url_str = getattr(img_val, "url")
                    if url_str:
                        att = process_image_source(url_str, client)
                        attachments.append(att)

    prompt = " ".join([t.strip() for t in text_parts if t.strip()])
    if not prompt and attachments:
        prompt = "Deskripsikan dan analisis gambar ini secara detail."

    return prompt, attachments


def sse_event_stream(
    req: ChatCompletionRequest,
    token: str,
    conv_id: str,
    session_id: str,
    parent_msg_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None
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

    try:
        for event in client.stream_chat(
            prompt=prompt,
            model=req.model or "gpt-5-6-thinking",
            parent_message_id=parent_msg_id or "client-created-root",
            conversation_id=conv_id_for_upstream,
            thinking=effective_thinking,
            attachments=attachments
        ):
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

    # Extract user prompt and any attached images
    last_user_prompt, attachments = extract_prompt_and_attachments(req.messages, client)

    if not last_user_prompt and not attachments:
        raise HTTPException(status_code=400, detail="No user message found in request")

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
                attachments=attachments
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
                attachments=attachments
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
                    attachments=attachments
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
