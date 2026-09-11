import json
from unittest.mock import MagicMock
from app.core.client import ChatGPTUpstreamClient

def test_sse_streaming_delta_formats():
    """
    Verifies that all three SSE streaming delta formats emitted by upstream ChatGPT are parsed:
    Format 1: {"p": "/message/content/parts/0", "o": "append", "v": "..."}
    Format 2: {"v": "..."} (standalone delta token)
    Format 3: {"p": "", "o": "patch", "v": [{"p": "/message/content/parts/0", "o": "append", "v": "..."}]}
    """
    raw_sse_lines = [
        # Metadata chunk
        'data: {"conversation_id": "test-conv-123", "message_id": "test-msg-001"}',
        # Format 1: Initial burst
        'data: {"p": "/message/content/parts/0", "o": "append", "v": "Gravitasi adalah gaya yang"}',
        # Format 2: Standalone text delta frames (dropped in buggy versions)
        'data: {"v": " menarik benda"}',
        'data: {"v": " ke arah"}',
        # Format 3: Patch array
        'data: {"p": "", "o": "patch", "v": [{"p": "/message/content/parts/0", "o": "append", "v": " pusat massa."}]}',
        # Reasoning recap
        'data: {"v": {"message": {"id": "msg-final-002", "content": {"content_type": "reasoning_recap", "content": "Definisi fisika sederhana."}}}}',
        # Done event
        'data: [DONE]'
    ]

    client = ChatGPTUpstreamClient(token="mock_token")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [line.encode("utf-8") for line in raw_sse_lines]

    client.get_chat_requirements = MagicMock()
    client.prepare_conversation = MagicMock(return_value="mock_conduit")
    client.session.post = MagicMock(return_value=mock_resp)

    events = list(client.stream_chat(prompt="Jelaskan gravitasi dalam 1 kalimat"))

    text_deltas = [e["content"] for e in events if e.get("type") == "text"]
    full_text = "".join(text_deltas)
    expected_text = "Gravitasi adalah gaya yang menarik benda ke arah pusat massa."

    print(f"Parsed text: '{full_text}'")
    assert full_text == expected_text, f"Mismatch!\nExpected: '{expected_text}'\nGot:      '{full_text}'"

    reasoning_deltas = [e["reasoning"] for e in events if e.get("type") == "reasoning"]
    assert len(reasoning_deltas) == 1, "Missing reasoning recap"
    assert reasoning_deltas[0] == "Definisi fisika sederhana."

    print("TEST PASS: All 3 SSE text delta streaming formats successfully parsed without truncation!")

if __name__ == "__main__":
    test_sse_streaming_delta_formats()
