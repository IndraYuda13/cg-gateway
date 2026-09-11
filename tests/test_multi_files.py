import io
import base64
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from PIL import Image

from app.core.client import (
    ChatGPTUpstreamClient,
    guess_mime_type,
    inspect_image
)
from app.api.schemas import (
    MessageItem,
    ContentPart,
    FileUrlDetail,
    ImageUrlDetail,
    ChatCompletionRequest
)
from app.api.routes import (
    process_attachment_source,
    extract_prompt_and_attachments
)
from client.chatgpt_client import (
    build_user_content,
    encode_attachment_source,
    ChatGPTCLIClient
)


def create_sample_png(width: int = 50, height: int = 50, color: str = "red") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_guess_mime_type():
    assert guess_mime_type("test.pdf") == "application/pdf"
    assert guess_mime_type("data.csv") == "text/csv"
    assert guess_mime_type("notes.txt") == "text/plain"
    assert guess_mime_type("document.docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert guess_mime_type("sheet.xlsx") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert guess_mime_type("image.png") == "image/png"
    assert guess_mime_type("photo.jpeg") == "image/jpeg"
    assert guess_mime_type("config.json") == "application/json"
    assert guess_mime_type("unknown_ext.xyz123") == "application/octet-stream"


def test_document_upload_3_phase_lifecycle_with_my_files():
    client = ChatGPTUpstreamClient(token="mock_token")
    pdf_bytes = b"%PDF-1.4 sample pdf content for testing document indexing"

    mock_init_resp = MagicMock()
    mock_init_resp.status_code = 200
    mock_init_resp.json.return_value = {
        "status": "success",
        "upload_url": "https://azure.storage.test/blob-container/laporan.pdf?sas=xyz",
        "file_id": "file-pdf-8899"
    }

    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 201

    mock_done_resp = MagicMock()
    mock_done_resp.status_code = 200
    mock_done_resp.json.return_value = {
        "status": "success",
        "download_url": "https://azure.storage.test/download/laporan.pdf"
    }

    with patch.object(client.session, "post", side_effect=[mock_init_resp, mock_done_resp]) as mock_post, \
         patch("app.core.client.requests.put", return_value=mock_put_resp) as mock_put:

        # 1. Document upload with auto-detected use_case="my_files"
        result = client.upload_file(
            file_bytes=pdf_bytes,
            file_name="laporan.pdf"
        )

        assert result["file_id"] == "file-pdf-8899"
        assert result["file_name"] == "laporan.pdf"
        assert result["mime_type"] == "application/pdf"
        assert result["use_case"] == "my_files"
        assert result["width"] is None
        assert result["height"] is None

        # Verify initiation payload
        init_call = mock_post.call_args_list[0]
        assert init_call[1]["json"]["use_case"] == "my_files"
        assert init_call[1]["json"]["file_name"] == "laporan.pdf"

        # Verify Azure Blob Storage PUT headers
        mock_put.assert_called_once()
        put_headers = mock_put.call_args[1]["headers"]
        assert put_headers["Content-Type"] == "application/pdf"
        assert put_headers["x-ms-blob-type"] == "BlockBlob"

        # Verify finalize endpoint called
        fin_call = mock_post.call_args_list[1]
        assert "file-pdf-8899/uploaded" in fin_call[0][0]


def test_upload_file_auto_detection_image_vs_document():
    client = ChatGPTUpstreamClient(token="mock_token")

    mock_init_resp = MagicMock()
    mock_init_resp.status_code = 200
    mock_init_resp.json.return_value = {
        "upload_url": "https://azure.storage.test/blob",
        "file_id": "file-mock-123"
    }

    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 200

    mock_done_resp = MagicMock()
    mock_done_resp.status_code = 200
    mock_done_resp.json.return_value = {"download_url": "https://azure.storage.test/file"}

    with patch.object(client.session, "post", side_effect=[mock_init_resp, mock_done_resp, mock_init_resp, mock_done_resp]), \
         patch("app.core.client.requests.put", return_value=mock_put_resp):

        # Image -> use_case="multimodal"
        png_bytes = create_sample_png(80, 80, "blue")
        img_res = client.upload_file(file_bytes=png_bytes, file_name="gambar.png")
        assert img_res["use_case"] == "multimodal"
        assert img_res["width"] == 80
        assert img_res["height"] == 80

        # TXT Document -> use_case="my_files"
        txt_bytes = b"Catatan rahasia penting 12345"
        txt_res = client.upload_file(file_bytes=txt_bytes, file_name="rahasia.txt")
        assert txt_res["use_case"] == "my_files"
        assert txt_res["mime_type"] == "text/plain"


def test_mixed_attachments_and_incognito_upstream_payload():
    client = ChatGPTUpstreamClient(token="mock_token")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [
        b'data: {"conversation_id": "test-cid", "message_id": "test-mid"}',
        b'data: {"p": "/message/content/parts/0", "o": "append", "v": "Hasil analisis"}',
        b'data: [DONE]'
    ]

    client.get_chat_requirements = MagicMock()
    client.prepare_conversation = MagicMock(return_value="mock_conduit")
    client.session.post = MagicMock(return_value=mock_resp)

    # Mixed attachments: 2 images + 1 document
    attachments = [
        {
            "file_id": "file-img-1",
            "file_name": "chart1.png",
            "size_bytes": 1000,
            "mime_type": "image/png",
            "width": 120,
            "height": 90,
            "use_case": "multimodal"
        },
        {
            "file_id": "file-doc-1",
            "file_name": "data.csv",
            "size_bytes": 500,
            "mime_type": "text/csv",
            "use_case": "my_files"
        },
        {
            "file_id": "file-img-2",
            "file_name": "chart2.jpg",
            "size_bytes": 1500,
            "mime_type": "image/jpeg",
            "width": 200,
            "height": 150,
            "use_case": "multimodal"
        }
    ]

    events = list(client.stream_chat(
        prompt="Analisis chart dan data CSV ini",
        attachments=attachments,
        history_and_training_disabled=True
    ))

    # Verify prepare_conversation called with history_and_training_disabled=True
    client.prepare_conversation.assert_called_once()
    assert client.prepare_conversation.call_args[1]["history_and_training_disabled"] is True

    # Verify conversation post payload
    conv_call = client.session.post.call_args
    assert conv_call is not None
    post_payload = conv_call[1]["json"]

    # In incognito mode, history_and_training_disabled must be True in upstream payload
    assert post_payload["history_and_training_disabled"] is True

    user_msg = post_payload["messages"][0]
    msg_content = user_msg["content"]
    assert msg_content["content_type"] == "multimodal_text"
    parts = msg_content["parts"]

    # parts must contain:
    # 1. image_asset_pointer for chart1
    # 2. image_asset_pointer for chart2
    # 3. text prompt
    # Document data.csv must NOT be in parts as image_asset_pointer
    assert len(parts) == 3
    assert parts[0]["content_type"] == "image_asset_pointer"
    assert parts[0]["asset_pointer"] == "file-service://file-img-1"
    assert parts[0]["width"] == 120
    assert parts[0]["height"] == 90

    assert parts[1]["content_type"] == "image_asset_pointer"
    assert parts[1]["asset_pointer"] == "file-service://file-img-2"
    assert parts[1]["width"] == 200
    assert parts[1]["height"] == 150

    assert parts[2] == "Analisis chart dan data CSV ini"

    # All 3 attachments must be registered in metadata.attachments
    meta_attachments = user_msg["metadata"]["attachments"]
    assert len(meta_attachments) == 3
    assert meta_attachments[0]["id"] == "file-img-1"
    assert meta_attachments[0]["name"] == "chart1.png"
    assert meta_attachments[1]["id"] == "file-doc-1"
    assert meta_attachments[1]["name"] == "data.csv"
    assert meta_attachments[1]["mime_type"] == "text/csv"
    assert meta_attachments[2]["id"] == "file-img-2"
    assert meta_attachments[2]["name"] == "chart2.jpg"


def test_incognito_disabled_propagation():
    client = ChatGPTUpstreamClient(token="mock_token")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [b'data: [DONE]']

    client.get_chat_requirements = MagicMock()
    client.prepare_conversation = MagicMock(return_value="mock_conduit")
    client.session.post = MagicMock(return_value=mock_resp)

    list(client.stream_chat(
        prompt="Halo",
        history_and_training_disabled=False
    ))

    # Verify prepare_conversation called with history_and_training_disabled=False
    assert client.prepare_conversation.call_args[1]["history_and_training_disabled"] is False

    # Verify conversation payload has history_and_training_disabled=False
    conv_call = client.session.post.call_args
    assert conv_call[1]["json"]["history_and_training_disabled"] is False


def test_extract_prompt_and_attachments_multi_files():
    client = ChatGPTUpstreamClient(token="mock_token")
    client.upload_file = MagicMock(side_effect=[
        {"file_id": "fid-pdf", "file_name": "doc.pdf", "mime_type": "application/pdf", "use_case": "my_files"},
        {"file_id": "fid-img", "file_name": "photo.png", "mime_type": "image/png", "use_case": "multimodal"}
    ])

    pdf_b64 = base64.b64encode(b"%PDF-1.4 sample").decode("utf-8")
    img_b64 = base64.b64encode(create_sample_png(20, 20, "green")).decode("utf-8")

    messages = [
        MessageItem(
            role="user",
            content=[
                {"type": "text", "text": "Bandingkan PDF dan gambar ini"},
                {"type": "file_url", "file_url": {"url": f"data:application/pdf;name=doc.pdf;base64,{pdf_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        )
    ]

    prompt, attachments = extract_prompt_and_attachments(messages, client)
    assert prompt == "Bandingkan PDF dan gambar ini"
    assert len(attachments) == 2
    assert attachments[0]["file_id"] == "fid-pdf"
    assert attachments[1]["file_id"] == "fid-img"
    assert client.upload_file.call_count == 2


def test_schema_chat_completion_request_incognito_default():
    req = ChatCompletionRequest(
        messages=[MessageItem(role="user", content="Test prompt")]
    )
    # Default incognito mode must be True
    assert req.history_and_training_disabled is True

    # Explicit False
    req_no_incog = ChatCompletionRequest(
        messages=[MessageItem(role="user", content="Test prompt")],
        history_and_training_disabled=False
    )
    assert req_no_incog.history_and_training_disabled is False


def test_client_build_user_content_multi(tmp_path):
    # 1. Plain text
    c1 = build_user_content("Hello")
    assert c1 == "Hello"

    # 2. Multi-attachments with real files
    img_file = tmp_path / "test.png"
    img_file.write_bytes(create_sample_png(10, 10, "black"))
    doc_file = tmp_path / "doc.pdf"
    doc_file.write_text("dummy pdf")
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2")

    c2 = build_user_content("Prompt", images=[str(img_file)], files=[str(doc_file), str(csv_file)])
    assert isinstance(c2, list)
    assert len(c2) == 4
    assert c2[0]["type"] == "text"
    assert c2[0]["text"] == "Prompt"
    assert c2[1]["type"] == "image_url"
    assert c2[2]["type"] == "file_url"
    assert c2[3]["type"] == "file_url"
