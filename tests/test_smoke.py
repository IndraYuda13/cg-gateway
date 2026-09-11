import sys
import time
from app.core.client import ChatGPTUpstreamClient

def run_tests():
    print("=== [1/4] Testing Sentinel Handshake (PoW + Turnstile VM) ===")
    client = ChatGPTUpstreamClient()
    req = client.get_chat_requirements(force_refresh=True)
    assert req.token, "Requirements token missing"
    assert req.proof_token, "Proof token missing"
    print(f"PASS: Handshake OK (Token: {req.token[:20]}...)")

    print("\n=== [2/4] Testing Model Listing ===")
    models = client.list_models()
    assert len(models) > 0, "No models returned"
    model_ids = [m["id"] for m in models]
    print(f"PASS: {len(models)} models available: {model_ids[:5]}...")

    print("\n=== [3/4] Testing Single-Turn Stream Chat (model: gpt-5-6) ===")
    prompt = "Ping test. Balas: 'ONLINE_OK'."
    text_buf = []
    for event in client.stream_chat(prompt=prompt, model="gpt-5-6"):
        if event.get("type") == "text":
            text_buf.append(event.get("content", ""))
    resp_text = "".join(text_buf)
    print(f"PASS: Upstream response received: {resp_text.strip()}")
    assert len(resp_text) > 0, "Empty response received"

    print("\n=== [4/4] Testing Multi-Turn Session Continuity ===")
    prompt_1 = "Kunci rahasia adalah: ALPHA-998. Balas 'SIAAP'."
    res_1 = client.chat_completion(prompt=prompt_1, model="gpt-5-6")
    conv_id = res_1.get("conversation_id")
    msg_id = res_1.get("message_id")
    assert conv_id and msg_id, "Missing conv_id or msg_id from turn 1"

    prompt_2 = "Apa kunci rahasia yang tadi saya berikan?"
    res_2 = client.chat_completion(
        prompt=prompt_2,
        model="gpt-5-6",
        parent_message_id=msg_id,
        conversation_id=conv_id
    )
    turn_2_text = res_2.get("content", "")
    print(f"Turn 2 Response: {turn_2_text.strip()}")
    assert "ALPHA-998" in turn_2_text, "Failed to recall key from turn 1"
    print("PASS: Multi-turn session memory verified!")

    print("\n=======================================================")
    print("  ALL UPSTREAM CORE TESTS PASSED (4/4) - 100% SUCCESS  ")
    print("=======================================================")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
