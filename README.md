# ⚡ cg-gateway

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

High-performance, lightweight OpenAI-compatible reverse API gateway and interactive terminal client for upstream conversational endpoints (CG edition).

Engineered with sub-millisecond cryptographic challenge solvers, an intelligent Least-Recently-Used (LRU) session pool orchestrator that prevents workspace conversation clutter, and native streaming support for both fast responses and deep reasoning tokens (`reasoning_content`).

---

## 🌟 Highlights

- ⚡ **High-Speed Challenge Resolution**:
  - Pure Python SHA3-512 Proof-of-Work (PoW) solver matching dynamic difficulty challenges in ~0.003s.
  - Fully self-contained Turnstile Bytecode Virtual Machine (VM) emulator executing challenges in ~0.01s.
  - Two-stage sentinel handshake (`/prepare` and `/finalize`) with thread-safe cached token renewal.
- 🔒 **Smart Session Pool & Multi-Turn State Management**:
  - **Context Fingerprinting**: Groups sequential turns deterministically via SHA-256 root prompt digests.
  - **Zero Workspace Clutter**: Eliminates orphaned chat sessions upstream by strictly bounding active conversation pools and triggering background deletion of evicted threads.
  - **Auto-Healing Resilience**: Automatically detects invalidated or expired remote sessions and creates seamless replacements.
- 🧠 **Frontier Model & Deep Reasoning Stream**:
  - Full support for `gpt-5-6`, `gpt-5-6-thinking`, `gpt-5-5`, `gpt-5-5-thinking`, `gpt-5-6-pro`, `gpt-6-pro`, `o3-pro`, and `gpt-4o`.
  - Isolates and streams reasoning tokens (`reasoning_content`) in real-time alongside final completion deltas.
- 🔌 **Drop-in OpenAI SDK Compatibility**:
  - Implements `/v1/chat/completions` (Server-Sent Events streaming and buffered JSON).
  - Compatible with official `openai` Python and Node.js SDKs, LangChain, LobeChat, LibreChat, NextChat, and OpenWebUI.
- 💻 **Zero-Dependency CLI & REPL**:
  - Standalone terminal client powered 100% by the Python Standard Library (`urllib`, `json`, `argparse`).
  - Interactive REPL with syntax coloring, reasoning toggle (`--no-think`), live model switching (`/model`), and session management (`/new`).
- 🐳 **Production Packaging**:
  - Pre-configured Docker, Docker Compose, and Systemd deployment service units.

---

## 📐 Architecture

```text
[ Client Applications / SDKs / CLI ]
                │
                ▼ (Standard OpenAI REST / SSE)
    [ Reverse Proxy / CF Tunnel ]
                │
                ▼ (Port :8560)
       [ cg-gateway (FastAPI) ]
                │
   ┌────────────┴────────────┐
   ▼                         ▼
[ app/api ]            [ app/core ]
  - /health              - SmartSessionPool (LRU + Background Eviction)
  - /v1/models           - PoW (SHA3-512) & Turnstile VM Solver
  - /v1/chat/completions - ChatGPTUpstreamClient (curl_cffi TLS impersonation)
                             │
                             ▼ (HTTP/2 Server-Sent Events)
                  [ Upstream Web Endpoint ]
                    ├─ POST /backend-api/sentinel/chat-requirements/prepare
                    ├─ POST /backend-api/sentinel/chat-requirements/finalize
                    ├─ POST /backend-api/f/conversation/prepare
                    ├─ POST /backend-api/f/conversation (SSE stream)
                    └─ PATCH /backend-api/conversation/{id} (cleanup)
```

---

## 📂 Repository Structure

```text
cg-gateway/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment configuration & credential manager
│   ├── main.py                # FastAPI application & middleware initialization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /health, /v1/models, /v1/chat/completions
│   │   └── schemas.py         # OpenAI Pydantic request/response schemas
│   └── core/
│       ├── __init__.py
│       ├── client.py          # ChatGPTUpstreamClient & SSE parser
│       ├── pow.py             # SHA3-512 PoW & Turnstile VM emulator
│       └── session.py         # SmartSessionPool LRU state manager
├── client/
│   ├── __init__.py
│   ├── cg_cli.py              # Zero-dependency interactive CLI & REPL
│   └── chatgpt_client.py      # Standalone verification tool & client
├── data/
│   └── .gitkeep               # Directory placeholder (credentials & sessions ignored)
├── systemd/
│   └── cg-gateway.service     # Systemd production service definition
├── tests/
│   ├── __init__.py
│   ├── test_api_multi_turn.py # Multi-turn API integration tests
│   ├── test_cli_client.py     # CLI client unit tests
│   ├── test_session_pool.py   # SmartSessionPool unit tests
│   ├── test_smoke.py          # Smoke tests
│   └── test_sse_parsing.py    # SSE stream decoder tests
├── .env.example               # Environment variables template
├── .gitignore                 # Strict git ignore definitions
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # Container orchestration configuration
├── LICENSE                    # MIT License
├── main.py                    # Gateway launch entry point
├── pyproject.toml             # Project metadata & build configuration
├── requirements.txt           # Python package dependencies
├── run_cli.py                 # CLI launcher shortcut
└── server.py                  # Uvicorn server launcher
```

---

## 🚀 Quickstart

### 1. Interactive Terminal Client (Zero External Dependencies)

The included CLI uses only the Python Standard Library (`urllib`, `argparse`, `json`):

```bash
# Clone the repository
git clone https://github.com/IndraYuda13/cg-gateway.git
cd cg-gateway

# Launch interactive terminal REPL
python3 run_cli.py

# Or execute a single prompt directly
python3 run_cli.py "Explain event-driven architecture in two sentences."

# Disable extended reasoning for instant response
python3 run_cli.py --no-think "What is the capital of Indonesia?"
```

Inside the interactive REPL:
- Enter any prompt to stream response with real-time reasoning visualization.
- `/new` or `/reset`: Clears conversation history and starts a fresh session.
- `/model <slug>`: Dynamically switch target model (e.g., `/model gpt-5-6`).
- `/exit` or `/quit`: Exits the client.

---

### 2. Self-Hosting (Local Machine or VPS)

#### Prerequisites
- Python 3.10+
- `pip` package manager

#### Setup & Execution
```bash
# 1. Clone repository
git clone https://github.com/IndraYuda13/cg-gateway.git
cd cg-gateway

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and configure your credentials (see Upstream Credential Acquisition)

# 5. Start gateway server
python3 server.py
```
The gateway will start listening on `http://0.0.0.0:8560`.

---

### 3. Docker Deployment

Deploy with Docker Compose:

```bash
# Build and launch container in background
docker compose up -d --build

# Inspect live logs
docker compose logs -f
```

To stop:
```bash
docker compose down
```

---

### 4. Production Systemd Service

For continuous background execution on Linux hosts:

```bash
# Copy systemd service unit
sudo cp systemd/cg-gateway.service /etc/systemd/system/

# Reload systemd daemon and enable service
sudo systemctl daemon-reload
sudo systemctl enable --now cg-gateway

# Verify service health & inspect logs
sudo systemctl status cg-gateway
sudo journalctl -u cg-gateway -f
```

---

## 🔑 Upstream Credential Acquisition

The gateway requires your active session authentication token and cookies from the upstream web service:

1. Open your desktop browser and log into your account.
2. Open Developer Tools (`F12` or `Ctrl + Shift + I`) and select the **Network** tab.
3. Send any message in the chat interface.
4. Filter requests by `conversation` or `chat-requirements`.
5. Under **Request Headers**, copy:
   - `authorization` token (e.g., `Bearer eyJhbGci...`)
   - `cookie` header string
6. Configure them either via `.env`:
   ```bash
   CG_TOKEN="eyJhbGciOi..."
   CG_COOKIES="__Secure-next-auth.session-token=...; ..."
   ```
   Or place them in `data/credentials.json`:
   ```json
   {
     "token": "eyJhbGciOi...",
     "cookies": "__Secure-next-auth.session-token=...",
     "account_id": "optional-workspace-uuid"
   }
   ```

*(Note: `data/credentials.json` is strictly ignored by git and never committed).*

---

## 📡 API Reference

### Health & Runtime Diagnostics
```bash
curl -s http://127.0.0.1:8560/health | jq
```

**Response:**
```json
{
  "status": "online",
  "service": "cg-gateway",
  "version": "1.0.0",
  "account_id": "00000000-0000-0000-0000-000000000000",
  "models": [
    "gpt-5-6",
    "gpt-5-6-thinking",
    "gpt-5-5",
    "gpt-5-5-thinking",
    "gpt-5-6-pro",
    "gpt-6-pro",
    "o3-pro",
    "gpt-4o",
    "gpt-4o-mini"
  ],
  "pool": {
    "active_conversations_count": 1,
    "max_pool_size": 10,
    "conversations": []
  },
  "account": {
    "name": "Workspace User",
    "email": "user@example.com",
    "plan": "Enterprise / Workspace"
  }
}
```

---

### Model Discovery
```bash
curl -s http://127.0.0.1:8560/v1/models | jq
```

---

### Chat Completions (Streaming SSE)
```bash
curl -N http://127.0.0.1:8560/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-6-thinking",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "stream": true
  }'
```

---

### Chat Completions (Buffered JSON)
```bash
curl -s http://127.0.0.1:8560/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-6-thinking",
    "messages": [
      {"role": "user", "content": "What is 15 * 18?"}
    ],
    "stream": false
  }' | jq
```

---

### Using Official OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8560/v1",
    api_key="lemon"  # Matches PROXY_API_KEY if configured
)

response = client.chat.completions.create(
    model="gpt-5-6-thinking",
    messages=[
        {"role": "user", "content": "Write a concise Python function to check prime numbers."}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    # Isolate deep reasoning stream if present
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

---

## 🧪 Verification & Testing

The repository includes a comprehensive unit and integration test suite:

```bash
# Run test suite
pytest tests/
```

All tests execute against mock fixtures and local logic, validating:
- Multi-turn conversation continuity and parent message chaining.
- CLI client streaming decoder and reasoning separation.
- SmartSessionPool LRU eviction and memory bounds.
- Upstream SSE protocol frame decoding.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
