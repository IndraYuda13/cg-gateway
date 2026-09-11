import time
import uuid
import json
from typing import Optional, Any, Generator
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import (
    DEFAULT_TOKEN,
    PROXY_API_KEY,
    ACCOUNT_ID,
    DEFAULT_MODELS
)
from app.core.session import smart_pool
from app.core.client import ChatGPTUpstreamClient
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


def sse_event_stream(
    req: ChatCompletionRequest,
    token: str,
    conv_id: str,
    session_id: str,
    parent_msg_id: str,
    prompt: str
) -> Generator[str, None, None]:
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    client = ChatGPTUpstreamClient(token=token)

    # Initial chunk with assistant role
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "session_id": session_id,
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
            thinking=effective_thinking
        ):
            e_type = event.get("type")

            if e_type == "text":
                content = event.get("content", "")
                if content:
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "session_id": last_conv_id or session_id,
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
                        "session_id": last_conv_id or session_id,
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
        err_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "session_id": session_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n[Error from upstream: {str(e)}]"},
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
        "session_id": last_conv_id or session_id,
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

    # Extract user prompt
    messages_dicts = [m.model_dump() for m in req.messages]
    last_user_prompt = ""
    for m in reversed(req.messages):
        if m.role == "user":
            c = m.content
            if isinstance(c, str):
                last_user_prompt = c
            elif isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        parts.append(p)
                last_user_prompt = " ".join(parts)
            break

    if not last_user_prompt:
        raise HTTPException(status_code=400, detail="No user message found in request")

    # Compute conversation identity and pool entry
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
                prompt=last_user_prompt
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

        completion_res = client.chat_completion(
            prompt=last_user_prompt,
            model=req.model or "gpt-5-6-thinking",
            parent_message_id=parent_id_to_use,
            conversation_id=conv_id_for_upstream,
            thinking=effective_thinking
        )

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

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model or "gpt-5-6-thinking",
            session_id=final_conv_id or session_id,
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
