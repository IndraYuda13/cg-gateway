import urllib.request
import json
import time

def test_api_multi_turn_continuity():
    base_url = "http://127.0.0.1:8560/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lemon"
    }

    # Turn 1: Establish fact in conversation
    pet_name = "Mochi-99"
    payload_1 = {
        "model": "gpt-5-6-thinking",
        "messages": [
            {"role": "user", "content": f"Nama kucing peliharaan saya adalah: {pet_name}. Balas hanya: 'OK'."}
        ]
    }

    req_1 = urllib.request.Request(base_url, data=json.dumps(payload_1).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req_1, timeout=60) as resp_1:
        res_data_1 = json.loads(resp_1.read().decode("utf-8"))

    session_id_1 = res_data_1.get("session_id")
    answer_1 = res_data_1["choices"][0]["message"]["content"]
    print(f"Turn 1 Answer: {answer_1.strip()}")
    print(f"Turn 1 Session ID: {session_id_1}")
    assert session_id_1, "No session_id returned in Turn 1"

    # Turn 2: Pass explicit session_id and ask for secret
    payload_2 = {
        "model": "gpt-5-6-thinking",
        "session_id": session_id_1,
        "messages": [
            {"role": "user", "content": "Siapa nama kucing peliharaan saya yang tadi?"}
        ]
    }

    req_2 = urllib.request.Request(base_url, data=json.dumps(payload_2).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req_2, timeout=60) as resp_2:
        res_data_2 = json.loads(resp_2.read().decode("utf-8"))

    session_id_2 = res_data_2.get("session_id")
    answer_2 = res_data_2["choices"][0]["message"]["content"]
    print(f"Turn 2 Answer: {answer_2.strip()}")
    print(f"Turn 2 Session ID: {session_id_2}")

    assert pet_name in answer_2, f"Expected {pet_name} in answer_2, got: {answer_2}"
    assert session_id_2 == session_id_1, f"Expected session_id to persist ({session_id_1}), got {session_id_2}"
    print(f"PASS: Multi-turn session continuity over HTTP API with session_id verified! ({pet_name} recalled)")

    # Turn 3: SSE Streaming Turn with explicit session_id
    payload_3 = {
        "model": "gpt-5-6-thinking",
        "session_id": session_id_2,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Kucing siapa Mochi-99 itu?"}
        ]
    }
    req_3 = urllib.request.Request(base_url, data=json.dumps(payload_3).encode("utf-8"), headers=headers)
    stream_chunks = []
    stream_sess_id = None
    with urllib.request.urlopen(req_3, timeout=60) as resp_3:
        for line in resp_3:
            line_str = line.decode("utf-8", errors="ignore").strip()
            if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                try:
                    c = json.loads(line_str[6:])
                    s_id = c.get("session_id")
                    if s_id:
                        stream_sess_id = s_id
                    choices = c.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            stream_chunks.append(delta["content"])
                except Exception:
                    pass

    stream_answer = "".join(stream_chunks)
    print(f"Turn 3 (Stream) Answer: {stream_answer.strip()}")
    print(f"Turn 3 Session ID: {stream_sess_id}")
    assert stream_sess_id == session_id_1, f"Expected stream session_id {session_id_1}, got {stream_sess_id}"
    assert len(stream_answer) > 0, "Empty stream answer received"
    print("PASS: Turn 3 streaming with session_id continuity verified!")

if __name__ == "__main__":
    test_api_multi_turn_continuity()
