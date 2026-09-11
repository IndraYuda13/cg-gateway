import os
import time
import json
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Generator, Tuple
from curl_cffi import requests

from app.config import (
    UPSTREAM_BASE_URL,
    ACCOUNT_ID,
    DEFAULT_TOKEN,
    DEFAULT_COOKIES,
    USER_AGENT,
    DEFAULT_MODELS,
    DEBUG
)
from app.core.pow import (
    build_legacy_requirements_token,
    build_proof_token,
    solve_turnstile_token
)


def inspect_image(data: bytes) -> Tuple[int, int, str]:
    """
    Inspects image bytes and returns (width, height, mime_type).
    Uses PIL with graceful binary header fallback.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        fmt = (img.format or "").upper()
        mime_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
            "GIF": "image/gif",
            "BMP": "image/bmp",
            "TIFF": "image/tiff"
        }
        mime = mime_map.get(fmt, "image/jpeg")
        return w, h, mime
    except Exception:
        pass

    # Header-based fallback
    mime = "image/jpeg"
    w, h = 1024, 1024
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
        if len(data) >= 24:
            import struct
            w, h = struct.unpack(">II", data[16:24])
    elif data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        mime = "image/gif"
        if len(data) >= 10:
            import struct
            w, h = struct.unpack("<HH", data[6:10])
    elif data.startswith(b"\xff\xd8"):
        mime = "image/jpeg"

    return w, h, mime


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


class ChatRequirements:
    def __init__(
        self,
        token: str,
        proof_token: str = "",
        turnstile_token: str = "",
        expire_at: float = 0
    ):
        self.token = token
        self.proof_token = proof_token
        self.turnstile_token = turnstile_token
        self.expire_at = expire_at


class ChatGPTUpstreamClient:
    """
    Client for ChatGPT Web Upstream API.
    Handles Sentinel requirements handshake, Proof of Work, Turnstile VM,
    conduit token negotiation, and event-stream parsing.
    """
    def __init__(
        self,
        token: Optional[str] = None,
        cookies: Optional[str] = None,
        account_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        self.base_url = UPSTREAM_BASE_URL.rstrip("/")
        self.token = token or DEFAULT_TOKEN
        self.cookies = cookies or DEFAULT_COOKIES
        self.account_id = account_id or ACCOUNT_ID
        self.user_agent = user_agent or USER_AGENT

        self.session = requests.Session(impersonate="edge101")
        self._setup_headers()
        self._file_cache: Dict[str, Dict[str, Any]] = {}

    def upload_file(
        self,
        file_bytes: bytes,
        file_name: str,
        mime_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_case: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the 3-phase upload lifecycle to ChatGPT Upstream:
        1. POST /backend-api/files -> returns upload_url and file_id
        2. PUT file_bytes to upload_url (Azure Blob Storage)
        3. POST /backend-api/files/{file_id}/uploaded -> marks ready and gets download_url
        """
        if not mime_type:
            mime_type = guess_mime_type(file_name)

        if mime_type.startswith("image/"):
            if width is None or height is None:
                w, h, detected_mime = inspect_image(file_bytes)
                width = width or w
                height = height or h
                if detected_mime:
                    mime_type = detected_mime

        if not use_case:
            use_case = "multimodal" if mime_type.startswith("image/") else "my_files"

        # Deduplication check via SHA-256 and use_case
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_key = f"{file_hash}_{use_case}"
        if hasattr(self, "_file_cache") and cache_key in self._file_cache:
            return self._file_cache[cache_key]
        if hasattr(self, "_file_cache") and file_hash in self._file_cache:
            cached = self._file_cache[file_hash]
            if cached.get("use_case") == use_case:
                return cached

        # Step 1: Initiate upload
        init_url = f"{self.base_url}/backend-api/files"
        init_payload = {
            "file_name": file_name,
            "file_size": len(file_bytes),
            "use_case": use_case
        }
        init_resp = self.session.post(init_url, json=init_payload, timeout=30)
        if init_resp.status_code != 200:
            raise RuntimeError(
                f"File upload initiation failed (HTTP {init_resp.status_code}): {init_resp.text[:300]}"
            )

        init_data = init_resp.json()
        upload_url = init_data.get("upload_url")
        file_id = init_data.get("file_id")
        if not upload_url or not file_id:
            raise RuntimeError(f"Missing upload_url or file_id in upload initiation: {init_data}")

        # Step 2: PUT raw binary to upload_url (Azure Blob Storage)
        # Note: Do NOT send ChatGPT session cookies/auth headers to Azure Blob Storage
        put_headers = {
            "Content-Type": mime_type,
            "x-ms-blob-type": "BlockBlob",
            "x-ms-version": "2020-04-08"
        }
        put_resp = requests.put(
            upload_url,
            data=file_bytes,
            headers=put_headers,
            timeout=60,
            impersonate="edge101"
        )
        if put_resp.status_code not in (200, 201):
            raise RuntimeError(
                f"File binary upload failed (HTTP {put_resp.status_code}): {put_resp.text[:300]}"
            )

        # Step 3: Finalize uploaded file
        uploaded_url = f"{self.base_url}/backend-api/files/{file_id}/uploaded"
        uploaded_resp = self.session.post(uploaded_url, json={}, timeout=30)
        if uploaded_resp.status_code != 200:
            raise RuntimeError(
                f"File upload finalize failed (HTTP {uploaded_resp.status_code}): {uploaded_resp.text[:300]}"
            )

        uploaded_data = uploaded_resp.json()
        download_url = uploaded_data.get("download_url")

        result = {
            "file_id": file_id,
            "file_name": file_name,
            "size_bytes": len(file_bytes),
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "use_case": use_case,
            "download_url": download_url
        }

        self._file_cache[cache_key] = result
        self._file_cache[file_hash] = result
        return result

    def _setup_headers(self) -> None:
        headers = {
            "User-Agent": self.user_agent,
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "OAI-Language": "en-US",
            "Sec-Ch-Ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Microsoft Edge";v="152"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.account_id:
            headers["ChatGPT-Account-Id"] = self.account_id
        if self.cookies:
            headers["Cookie"] = self.cookies

        self.session.headers.update(headers)

    def get_chat_requirements(self, force_refresh: bool = False) -> ChatRequirements:
        """
        Executes the 2-step Sentinel handshake (prepare & finalize) with PoW & Turnstile solving.
        Sentinel requirement tokens are single-use per conversation turn.
        """
        p_token = build_legacy_requirements_token(self.user_agent)
        prepare_url = f"{self.base_url}/backend-api/sentinel/chat-requirements/prepare"
        
        prep_resp = self.session.post(
            prepare_url,
            json={"p": p_token},
            timeout=20
        )
        if prep_resp.status_code != 200:
            raise RuntimeError(
                f"Sentinel prepare failed (HTTP {prep_resp.status_code}): {prep_resp.text[:300]}"
            )

        prep_data = prep_resp.json()
        prep_token = prep_data.get("prepare_token", "")

        # Step 1: Proof of Work solver
        proof_token = ""
        pow_info = prep_data.get("proofofwork") or {}
        if pow_info.get("required"):
            seed = pow_info.get("seed", "")
            difficulty = pow_info.get("difficulty", "")
            proof_token = build_proof_token(seed, difficulty, self.user_agent)

        # Step 2: Turnstile Bytecode VM solver
        turnstile_token = ""
        tt_info = prep_data.get("turnstile") or {}
        if tt_info.get("required") and tt_info.get("dx"):
            turnstile_token = solve_turnstile_token(tt_info["dx"], p_token) or ""

        # Step 3: Finalize requirements
        finalize_url = f"{self.base_url}/backend-api/sentinel/chat-requirements/finalize"
        fin_payload = {
            "prepare_token": prep_token,
            "proof_token": proof_token,
            "turnstile_token": turnstile_token
        }

        fin_resp = self.session.post(
            finalize_url,
            json=fin_payload,
            timeout=20
        )
        if fin_resp.status_code != 200:
            raise RuntimeError(
                f"Sentinel finalize failed (HTTP {fin_resp.status_code}): {fin_resp.text[:300]}"
            )

        fin_data = fin_resp.json()
        req_token = fin_data.get("token")
        if not req_token:
            raise RuntimeError(f"No token received from Sentinel finalize: {fin_data}")

        expire_at = fin_data.get("expire_at", time.time() + 500)
        return ChatRequirements(
            token=req_token,
            proof_token=proof_token,
            turnstile_token=turnstile_token,
            expire_at=expire_at
        )

    def prepare_conversation(
        self,
        model: str,
        parent_message_id: str = "client-created-root",
        prompt: str = "",
        conversation_id: Optional[str] = None,
        thinking: Optional[bool] = None,
        history_and_training_disabled: bool = True
    ) -> str:
        """
        Prepares conversation context and retrieves the conduit token.
        """
        conv_prep_url = f"{self.base_url}/backend-api/f/conversation/prepare"
        headers = {"X-Conduit-Token": "no-token", "Content-Type": "application/json"}
        
        is_thinking = ("thinking" in model.lower() or model.lower() in ("o3-pro", "gpt-5-6-pro", "gpt-6-pro")) if thinking is None else bool(thinking)
        payload = {
            "action": "next",
            "parent_message_id": parent_message_id,
            "model": model,
            "history_and_training_disabled": history_and_training_disabled,
            "client_prepare_state": "success",
            "client_prepare_dispatch": "immediate",
            "client_prepare_source": "context_change",
            "timezone_offset_min": -420,
            "timezone": "Asia/Jakarta",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": [],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "app_name": "chatgpt.com",
                "has_web_push_capabilities": True,
                "web_push_notification_permission": "default"
            }
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if is_thinking:
            payload["thinking_effort"] = "extended"

        resp = self.session.post(
            conv_prep_url,
            headers=headers,
            json=payload,
            timeout=20
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Conversation prepare failed (HTTP {resp.status_code}): {resp.text[:300]}"
            )

        data = resp.json()
        conduit_token = data.get("conduit_token")
        if not conduit_token:
            raise RuntimeError(f"Missing conduit_token in conversation prepare: {data}")
        return conduit_token

    def stream_chat(
        self,
        prompt: str,
        model: str = "gpt-5-6-thinking",
        parent_message_id: str = "client-created-root",
        conversation_id: Optional[str] = None,
        thinking: Optional[bool] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        history_and_training_disabled: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Main streaming chat generator.
        Yields structured SSE event dictionaries:
          - {"type": "text", "content": delta_text}
          - {"type": "reasoning", "reasoning": reasoning_delta}
          - {"type": "meta", "conversation_id": cid, "message_id": mid}
          - {"type": "done"}
        """
        requirements = self.get_chat_requirements()
        conduit_token = self.prepare_conversation(
            model=model,
            parent_message_id=parent_message_id,
            prompt=prompt,
            conversation_id=conversation_id,
            thinking=thinking,
            history_and_training_disabled=history_and_training_disabled
        )

        conv_url = f"{self.base_url}/backend-api/f/conversation"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Conduit-Token": conduit_token,
            "OpenAI-Sentinel-Proof-Token": requirements.proof_token,
            "OpenAI-Sentinel-Chat-Requirements-Token": requirements.token,
        }
        if requirements.turnstile_token:
            headers["OpenAI-Sentinel-Turnstile-Token"] = requirements.turnstile_token

        msg_id = str(uuid.uuid4())
        is_thinking = ("thinking" in model.lower() or model.lower() in ("o3-pro", "gpt-5-6-pro", "gpt-6-pro")) if thinking is None else bool(thinking)

        if attachments:
            parts: List[Any] = []
            attachments_meta: List[Dict[str, Any]] = []
            for att in attachments:
                fid = att.get("file_id") or att.get("id")
                fname = att.get("file_name") or att.get("name") or "file"
                size = att.get("size_bytes") or att.get("size") or 0
                mime = att.get("mime_type") or guess_mime_type(fname)
                w = att.get("width")
                h = att.get("height")
                use_case = att.get("use_case")

                # Image attachments get image_asset_pointer in parts
                is_image = (use_case == "multimodal") or mime.startswith("image/")
                if is_image:
                    img_part: Dict[str, Any] = {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": f"file-service://{fid}",
                        "size_bytes": size
                    }
                    if w is not None and h is not None:
                        img_part["width"] = w
                        img_part["height"] = h
                    parts.append(img_part)

                # All attachments (images & documents) get metadata entry
                meta_item: Dict[str, Any] = {
                    "id": fid,
                    "name": fname,
                    "size": size,
                    "mime_type": mime
                }
                if w is not None and h is not None:
                    meta_item["width"] = w
                    meta_item["height"] = h
                attachments_meta.append(meta_item)

            if prompt:
                parts.append(prompt)
            elif not any(isinstance(p, str) for p in parts):
                has_docs = any(not (a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal") for a in attachments)
                has_imgs = any(a.get("mime_type", "").startswith("image/") or a.get("use_case") == "multimodal" for a in attachments)
                if has_docs and has_imgs:
                    parts.append("Analisis dan jelaskan file dan gambar ini secara detail.")
                elif has_docs:
                    parts.append("Analisis dan ringkas dokumen ini secara detail.")
                else:
                    parts.append("Deskripsikan dan analisis gambar ini secara detail.")

            content_payload = {
                "content_type": "multimodal_text",
                "parts": parts
            }
            metadata_payload = {
                "selected_sources": [],
                "selected_github_repos": [],
                "selected_all_github_repos": False,
                "serialization_metadata": {"custom_symbol_offsets": []},
                "submission_mode": "manual_send",
                "attachments": attachments_meta
            }
        else:
            content_payload = {
                "content_type": "text",
                "parts": [prompt]
            }
            metadata_payload = {
                "selected_sources": [],
                "selected_github_repos": [],
                "selected_all_github_repos": False,
                "serialization_metadata": {"custom_symbol_offsets": []},
                "submission_mode": "manual_send"
            }

        payload: Dict[str, Any] = {
            "action": "next",
            "messages": [
                {
                    "id": msg_id,
                    "author": {"role": "user"},
                    "create_time": time.time(),
                    "content": content_payload,
                    "metadata": metadata_payload
                }
            ],
            "parent_message_id": parent_message_id,
            "model": model,
            "history_and_training_disabled": history_and_training_disabled,
            "client_prepare_state": "success",
            "timezone_offset_min": -420,
            "timezone": "Asia/Jakarta",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "system_hints": [],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "app_name": "chatgpt.com",
                "has_web_push_capabilities": True,
                "web_push_notification_permission": "default"
            }
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if is_thinking:
            payload["thinking_effort"] = "extended"

        resp = self.session.post(
            conv_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )

        if resp.status_code != 200:
            err_text = resp.text[:400]
            raise RuntimeError(f"Upstream conversation failed (HTTP {resp.status_code}): {err_text}")

        last_conv_id = conversation_id
        last_msg_id = None

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore").strip()
            if not line_str.startswith("data: "):
                continue

            raw_data = line_str[6:]
            if raw_data == "[DONE]":
                break

            try:
                data = json.loads(raw_data)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            # Extract conversation ID from tokens or markers
            if "conversation_id" in data and data["conversation_id"]:
                last_conv_id = data["conversation_id"]
            if "message_id" in data and data["message_id"]:
                last_msg_id = data["message_id"]
            if ("conversation_id" in data or "message_id" in data) and (last_conv_id or last_msg_id):
                yield {"type": "meta", "conversation_id": last_conv_id, "message_id": last_msg_id}

            # Handle delta append messages
            path = data.get("p", "")
            op = data.get("o", "")
            val = data.get("v")

            # Direct parts/0 append
            if path == "/message/content/parts/0" and op == "append" and isinstance(val, str):
                yield {"type": "text", "content": val}

            # Patch array with multiple operations
            elif (path == "" or path is None) and op == "patch" and isinstance(val, list):
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    i_path = item.get("p", "")
                    i_op = item.get("o", "")
                    i_val = item.get("v")
                    if i_path == "/message/content/parts/0" and i_op == "append" and isinstance(i_val, str):
                        yield {"type": "text", "content": i_val}
                    elif (i_path == "" or i_path is None) and (i_op == "" or i_op is None) and isinstance(i_val, str):
                        yield {"type": "text", "content": i_val}
                    elif ("reasoning" in str(i_path) or "thought" in str(i_path)) and isinstance(i_val, str):
                        yield {"type": "reasoning", "reasoning": i_val}

            # Standalone delta string token frame {"v": "..."} where p and o are omitted / empty / None
            elif (path == "" or path is None) and (op == "" or op is None) and isinstance(val, str):
                yield {"type": "text", "content": val}

            # Direct reasoning append if path matches
            elif ("reasoning" in str(path) or "thought" in str(path)) and isinstance(val, str):
                yield {"type": "reasoning", "reasoning": val}

            # Check for message structure updates
            elif isinstance(val, dict):
                msg_obj = val.get("message", {})
                if msg_obj:
                    mid = msg_obj.get("id")
                    if mid:
                        last_msg_id = mid
                        yield {"type": "meta", "conversation_id": last_conv_id, "message_id": last_msg_id}

                    content_obj = msg_obj.get("content", {})
                    c_type = content_obj.get("content_type")

                    # Handle reasoning / thought content
                    if c_type in ("reasoning_recap", "thought", "reasoning"):
                        recap_text = content_obj.get("content") or content_obj.get("text") or ""
                        if not recap_text and isinstance(content_obj.get("parts"), list):
                            recap_text = "".join(str(p) for p in content_obj.get("parts", []))
                        if recap_text:
                            yield {"type": "reasoning", "reasoning": recap_text}

        yield {"type": "done", "conversation_id": last_conv_id, "message_id": last_msg_id}

    def chat_completion(
        self,
        prompt: str,
        model: str = "gpt-5-6-thinking",
        parent_message_id: str = "client-created-root",
        conversation_id: Optional[str] = None,
        thinking: Optional[bool] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        history_and_training_disabled: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous non-streaming chat helper that consumes the stream and aggregates output.
        """
        full_text = []
        full_reasoning = []
        final_conv_id = conversation_id
        final_msg_id = None

        for event in self.stream_chat(
            prompt=prompt,
            model=model,
            parent_message_id=parent_message_id,
            conversation_id=conversation_id,
            thinking=thinking,
            attachments=attachments,
            history_and_training_disabled=history_and_training_disabled
        ):
            e_type = event.get("type")
            if e_type == "text":
                full_text.append(event.get("content", ""))
            elif e_type == "reasoning":
                full_reasoning.append(event.get("reasoning", ""))
            elif e_type == "meta":
                if event.get("conversation_id"):
                    final_conv_id = event["conversation_id"]
                if event.get("message_id"):
                    final_msg_id = event["message_id"]
            elif e_type == "done":
                if event.get("conversation_id"):
                    final_conv_id = event["conversation_id"]
                if event.get("message_id"):
                    final_msg_id = event["message_id"]

        return {
            "content": "".join(full_text),
            "reasoning_content": "".join(full_reasoning) if full_reasoning else None,
            "conversation_id": final_conv_id,
            "message_id": final_msg_id
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Fetches the models available on the account or returns the frontier defaults.
        """
        try:
            url = f"{self.base_url}/backend-api/models?iim=false&is_gizmo=false&supports_model_picker_upgrade_presets=true"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                raw_models = data.get("models", [])
                result = []
                seen = set()
                for m in raw_models:
                    slug = m.get("slug")
                    if slug and slug not in seen:
                        seen.add(slug)
                        result.append({
                            "id": slug,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "openai",
                            "root": slug,
                            "parent": None
                        })
                if result:
                    return result
        except Exception as e:
            if DEBUG:
                print(f"[ChatGPTUpstreamClient] Model list fetch warning: {e}")

        return [
            {
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": m["owned_by"],
                "root": m["id"],
                "parent": None
            }
            for m in DEFAULT_MODELS
        ]

    def get_user_profile(self) -> Dict[str, Any]:
        """
        Retrieves user information from the upstream account.
        """
        try:
            url = f"{self.base_url}/backend-api/me"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Soft-deletes a conversation upstream (is_visible: false).
        """
        try:
            url = f"{self.base_url}/backend-api/conversation/{conversation_id}"
            resp = self.session.patch(
                url,
                json={"is_visible": False},
                timeout=10
            )
            return resp.status_code == 200
        except Exception:
            return False

    def create_session(self) -> str:
        """
        Generates a new session UUID for pool management.
        """
        return str(uuid.uuid4())
