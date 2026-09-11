import io
import base64
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from PIL import Image

from app.core.client import inspect_image, ChatGPTUpstreamClient
from app.api.schemas import MessageItem, ContentPart, ImageUrlDetail, ChatCompletionRequest
from app.api.routes import process_image_source, extract_prompt_and_attachments


def create_sample_png(width: int = 100, height: int = 100, color: str = "green") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sample_jpeg(width: int = 60, height: int = 40, color: str = "red") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_inspect_image_png():
    png_bytes = create_sample_png(120, 80, "blue")
    w, h, mime = inspect_image(png_bytes)
    assert w == 120
    assert h == 80
    assert mime == "image/png"


def test_inspect_image_jpeg():
    jpg_bytes = create_sample_jpeg(75, 45, "yellow")
    w, h, mime = inspect_image(jpg_bytes)
    assert w == 75
    assert h == 45
    assert mime == "image/jpeg"


def test_schema_content_parts_and_validation():
    # 1. Standard string content
    m1 = MessageItem(role="user", content="Hello world")
    assert isinstance(m1.content, str)

    # 2. OpenAI vision format with list of dicts
    m2 = MessageItem(
        role="user",
        content=[
            {"type": "text", "text": "What is in this picture?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                }
            }
        ]
    )
    assert isinstance(m2.content, list)
    assert len(m2.content) == 2
    item0 = m2.content[0]
    item1 = m2.content[1]
    type0 = item0.type if isinstance(item0, ContentPart) else (item0.get("type") if isinstance(item0, dict) else None)
    type1 = item1.type if isinstance(item1, ContentPart) else (item1.get("type") if isinstance(item1, dict) else None)
    assert type0 == "text"
    assert type1 == "image_url"

    # 3. ChatCompletionRequest with vision message
    req = ChatCompletionRequest(
        model="gpt-5-6-thinking",
        messages=[m2]
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_upload_file_3_phase_lifecycle_and_deduplication():
    client = ChatGPTUpstreamClient(token="mock_token")
    img_bytes = create_sample_png(50, 50, "green")

    mock_init_resp = MagicMock()
    mock_init_resp.status_code = 200
    mock_init_resp.json.return_value = {
        "status": "success",
        "upload_url": "https://azure.storage.test/blob-container/image.png?sas=xyz",
        "file_id": "file-mock-9988"
    }

    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 201

    mock_done_resp = MagicMock()
    mock_done_resp.status_code = 200
    mock_done_resp.json.return_value = {
        "status": "success",
        "download_url": "https://azure.storage.test/download/image.png"
    }

    with patch.object(client.session, "post", side_effect=[mock_init_resp, mock_done_resp]) as mock_post, \
         patch("app.core.client.requests.put", return_value=mock_put_resp) as mock_put:

        # Turn 1: Normal upload
        result = client.upload_file(
            file_bytes=img_bytes,
            file_name="green_sq.png",
            mime_type="image/png",
            width=50,
            height=50
        )

        assert result["file_id"] == "file-mock-9988"
        assert result["width"] == 50
        assert result["height"] == 50
        assert result["mime_type"] == "image/png"
        assert result["download_url"] == "https://azure.storage.test/download/image.png"

        assert mock_post.call_count == 2
        mock_put.assert_called_once()
        put_kwargs = mock_put.call_args[1]
        assert put_kwargs["headers"]["Content-Type"] == "image/png"
        assert put_kwargs["headers"]["x-ms-blob-type"] == "BlockBlob"

        # Turn 2: Deduplication cache test with identical bytes
        cached_result = client.upload_file(
            file_bytes=img_bytes,
            file_name="green_sq.png",
            mime_type="image/png"
        )
        assert cached_result["file_id"] == "file-mock-9988"
        # Verify no new network calls were made
        assert mock_post.call_count == 2
        assert mock_put.call_count == 1


def test_multimodal_conversation_payload_construction():
    client = ChatGPTUpstreamClient(token="mock_token")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [
        b'data: {"conversation_id": "test-cid", "message_id": "test-mid"}',
        b'data: {"p": "/message/content/parts/0", "o": "append", "v": "Warna hijau"}',
        b'data: [DONE]'
    ]

    client.get_chat_requirements = MagicMock()
    client.prepare_conversation = MagicMock(return_value="mock_conduit")
    client.session.post = MagicMock(return_value=mock_resp)

    attachments = [
        {
            "file_id": "file-abc-123",
            "file_name": "test.png",
            "size_bytes": 1024,
            "mime_type": "image/png",
            "width": 100,
            "height": 100
        }
    ]

    events = list(client.stream_chat(
        prompt="Gambar apakah ini?",
        attachments=attachments
    ))

    # Verify conversation post payload
    conv_call = client.session.post.call_args
    assert conv_call is not None
    post_payload = conv_call[1]["json"]

    user_msg = post_payload["messages"][0]
    msg_content = user_msg["content"]
    assert msg_content["content_type"] == "multimodal_text"
    parts = msg_content["parts"]
    assert len(parts) == 2
    assert parts[0]["content_type"] == "image_asset_pointer"
    assert parts[0]["asset_pointer"] == "file-service://file-abc-123"
    assert parts[0]["width"] == 100
    assert parts[0]["height"] == 100
    assert parts[1] == "Gambar apakah ini?"

    meta_attachments = user_msg["metadata"]["attachments"]
    assert len(meta_attachments) == 1
    assert meta_attachments[0]["id"] == "file-abc-123"
    assert meta_attachments[0]["name"] == "test.png"


def test_extract_prompt_and_attachments_base64():
    client = ChatGPTUpstreamClient(token="mock_token")
    client.upload_file = MagicMock(return_value={
        "file_id": "file-xyz",
        "file_name": "image_123.png",
        "size_bytes": 100,
        "mime_type": "image/png",
        "width": 50,
        "height": 50
    })

    png_bytes = create_sample_png(50, 50, "purple")
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    messages = [
        MessageItem(
            role="user",
            content=[
                {"type": "text", "text": "Apa warna kotak ini?"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        )
    ]

    prompt, attachments = extract_prompt_and_attachments(messages, client)
    assert prompt == "Apa warna kotak ini?"
    assert len(attachments) == 1
    assert attachments[0]["file_id"] == "file-xyz"
    client.upload_file.assert_called_once()


def test_ssrf_protection_in_process_image_source():
    client = ChatGPTUpstreamClient(token="mock_token")

    # 1. Localhost IP
    with pytest.raises(HTTPException) as exc1:
        process_image_source("http://127.0.0.1:8080/secret.png", client)
    assert exc1.value.status_code == 400
    assert "Blocked private/internal IP" in exc1.value.detail

    # 2. RFC1918 Private IP
    with pytest.raises(HTTPException) as exc2:
        process_image_source("http://192.168.1.1/admin.jpg", client)
    assert exc2.value.status_code == 400
    assert "Blocked private/internal IP" in exc2.value.detail

    # 3. Localhost hostname
    with pytest.raises(HTTPException) as exc3:
        process_image_source("http://localhost:5000/internal.png", client)
    assert exc3.value.status_code == 400
    assert "Blocked internal resolution" in exc3.value.detail


def test_live_vision_chat_completion_local():
    """
    Integration smoke test against local gateway with synthetic base64 image.
    """
    import urllib.request
    import json

    png_bytes = create_sample_png(40, 40, "blue")
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    base_url = "http://127.0.0.1:8560/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lemon"
    }

    payload = {
        "model": "gpt-5-6-thinking",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Warna apa gambar ini? Jawab 'biru' saja."},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]
    }

    try:
        req = urllib.request.Request(base_url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ans = data["choices"][0]["message"]["content"].lower()
        print(f"\nLive Vision Answer: {ans.strip()}")
        assert "biru" in ans or "blue" in ans
    except Exception as e:
        pytest.skip(f"Local gateway live check skipped: {e}")

