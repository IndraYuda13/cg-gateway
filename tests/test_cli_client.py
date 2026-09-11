import pytest
import subprocess
import sys
from client.chatgpt_client import ChatGPTCLIClient, DEFAULT_LOCAL_URL

def test_client_headers():
    client = ChatGPTCLIClient(api_url="http://127.0.0.1:8560", api_key="lemon")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer lemon"
    assert headers["Content-Type"] == "application/json"

def test_client_health():
    client = ChatGPTCLIClient(api_url="http://127.0.0.1:8560", api_key="lemon")
    health = client.check_health()
    assert health.get("status") == "online"
    assert health.get("service") == "cg-gateway"
    assert "models" in health
    assert "gpt-5-6-thinking" in health["models"]

def test_cli_single_shot_execution():
    cmd = [
        sys.executable,
        "client/chatgpt_client.py",
        "--local",
        "Katakan 'PONG' saja."
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "PONG" in proc.stdout or "pong" in proc.stdout.lower()
