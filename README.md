# cg-gateway

**cg-gateway** is a high-performance, production-ready reverse API gateway that bridges OpenAI ChatGPT Web services with the standard OpenAI API SDK specification (`/v1/chat/completions`, `/v1/models`, `/health`).

It enables direct programmatic access to frontier models (including `gpt-5-6`, `gpt-5-6-thinking`, `gpt-5-5`, `gpt-5-5-thinking`, `gpt-5-6-pro`, `gpt-6-pro`, `o3-pro`, and `gpt-4o`) using standard OpenAI SDK clients, curl commands, or standalone terminal CLIs.

---

## Key Features

1. **OpenAI SDK Drop-in Compatibility**:
   - Implements `/v1/chat/completions` supporting both Server-Sent Events (`stream: true`) and buffered JSON responses (`stream: false`).
   - Parses and isolates `reasoning_content` (thinking tokens) alongside final assistant message content.
   - Standard `/v1/models` endpoint for automated model discovery.

2. **Full Sentinel & Anti-Bot Cryptographic Bypass**:
   - **Legacy P-Token Generator**: Generates synthetic browser device fingerprints and high-resolution timing telemetry.
   - **SHA3-512 Proof of Work (PoW) Solver**: High-speed pure Python solver matching dynamic difficulty challenges in ~0.003s.
   - **Turnstile Bytecode Virtual Machine Solver**: Fully contained bytecode deobfuscator and VM emulator that unpacks and executes challenge routines in ~0.01s.
   - **Two-Step Sentinel Handshake**: Automates `/prepare` and `/finalize` flows, maintaining an in-memory cached requirements token valid for ~9 minutes.
   - **Browser TLS & HTTP/2 Impersonation**: Uses `curl_cffi` with Edge/Chrome TLS profiles to bypass Cloudflare anti-bot checks.

3. **Smart Session Pool & Multi-Turn State Management**:
   - **Context Fingerprinting**: Automatically groups conversation turns using SHA-256 digests of the root user prompt.
   - **Parent Message Tracking**: Seamlessly tracks upstream `parent_message_id` and conversation IDs across sequential turns.
   - **LRU Session Eviction**: Strictly bounds pool size (default: 10 active conversations) and triggers upstream background deletion of evicted chats to prevent sidebar pollution.
   - **Auto-Healing**: Transparently creates replacement sessions if upstream sessions expire or encounter invalid states.

4. **Pure Python Terminal Client**:
   - Zero external dependencies CLI (`client/cg_cli.py` and `run_cli.py`).
   - Supports interactive REPL mode with colored streaming and reasoning toggle (`--no-think`).
   - Single-shot prompt CLI for shell scripts and automation pipelines.

---

## Architecture Overview

```
[ Client Application / OpenAI SDK / CLI ]
                   │
                   ▼ (HTTP / SSE on Port 8560)
         [ cg-gateway (FastAPI) ]
                   │
   ┌───────────────┼───────────────┐
   ▼               ▼               ▼
[ app/api ]  [ app/core ]    [ app/core ]
  Routes       Session Pool    PoW & Turnstile VM
                   │               │
                   └───────┬───────┘
                           ▼
              [ ChatGPTUpstreamClient ]
                           │ (curl_cffi edge101 TLS)
                           ▼
          [ Upstream: https://chatgpt.com ]
            ├─ POST /backend-api/sentinel/chat-requirements/prepare
            ├─ POST /backend-api/sentinel/chat-requirements/finalize
            ├─ POST /backend-api/f/conversation/prepare
            ├─ POST /backend-api/f/conversation (SSE stream)
            └─ PATCH /backend-api/conversation/{id} (LRU cleanup)
```

---

## Directory Structure

```text
/root/projects/cg-gateway/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration & credential loader
│   ├── main.py                # FastAPI app initialization & CORS
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /health, /v1/models, /v1/chat/completions
│   │   └── schemas.py         # Pydantic OpenAI schema models
│   └── core/
│       ├── __init__.py
│       ├── client.py          # ChatGPTUpstreamClient & SSE stream parser
│       ├── pow.py             # SHA3-512 PoW & Turnstile VM solver
│       └── session.py         # SmartSessionPool LRU state manager
├── client/
│   ├── __init__.py
│   └── cg_cli.py              # Standalone Python CLI & REPL
├── data/
│   ├── credentials.json       # Extracted account credentials & cookies
│   └── session_pool.json      # Persistent session cache
├── systemd/
│   └── cg-gateway.service     # Systemd production unit definition
├── .env.example
├── main.py                    # Gateway launch entrypoint
├── requirements.txt           # Project dependencies
├── run_cli.py                 # Convenience CLI launcher
├── server.py                  # Uvicorn server launcher
└── README.md
```

---

## Quick Start

### 1. Installation

Ensure Python 3.10+ is installed:

```bash
cd /root/projects/cg-gateway
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` if you need custom overrides:

```bash
cp .env.example .env
```

Environment variables:
- `PORT`: Gateway listening port (default: `8560`).
- `HOST`: Gateway listening host (default: `0.0.0.0`).
- `PROXY_API_KEY`: Optional security key required for gateway access.
- `MAX_SESSIONS`: Maximum active conversational sessions before LRU eviction (default: `10`).
- `CG_ACCOUNT_ID`: Target ChatGPT workspace / account UUID.

Credentials can be supplied via environment variables (`CG_TOKEN`, `CG_COOKIES`) or stored in `data/credentials.json`.

### 3. Running the Server

Direct execution:
```bash
python3 server.py
```

The gateway will start on `http://0.0.0.0:8560`.

---

## API Usage

### Health Check

```bash
curl -s http://127.0.0.1:8560/health | jq
```

Response:
```json
{
  "status": "online",
  "service": "cg-gateway",
  "version": "1.0.0",
  "account_id": "fb88ebc3-79ae-45ff-b8b1-2313efa099b4",
  "models": [
    "gpt-5-6",
    "gpt-5-6-thinking",
    "gpt-5-5",
    "gpt-5-5-thinking",
    "gpt-5-6-pro",
    "gpt-6-pro",
    "o3-pro",
    "gpt-4o"
  ],
  "pool": {
    "active_conversations_count": 1,
    "max_pool_size": 10,
    "conversations": [...]
  },
  "account": {
    "name": "csacsa",
    "email": "10482807+notisations@utc2eduvn.onmicrosoft.com",
    "plan": "Enterprise / Workspace"
  }
}
```

### List Models

```bash
curl -s http://127.0.0.1:8560/v1/models | jq
```

### Chat Completion (Streaming)

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

### Chat Completion (Non-Streaming)

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

### Using Official OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8560/v1",
    api_key="lemon"  # or your PROXY_API_KEY
)

response = client.chat.completions.create(
    model="gpt-5-6-thinking",
    messages=[
        {"role": "user", "content": "Write a Python function to check prime numbers."}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

---

## Terminal Client (CLI)

The included CLI requires only standard Python libraries:

### Single-shot Prompt:
```bash
python3 run_cli.py "Summarize Newton's laws of motion"
```

### Instant Mode (Disable Extended Reasoning):
```bash
python3 run_cli.py --no-think "What is the capital of Indonesia?"
```

### Interactive REPL Mode:
```bash
python3 run_cli.py
```

Inside the interactive REPL:
- Type your prompt and press Enter.
- `/new` or `/reset`: Clears multi-turn history and starts a fresh session.
- `/model <slug>`: Switches target model on the fly.
- `/exit`: Exits the client.

---

## Systemd Service Management

To deploy `cg-gateway` as a background system daemon:

1. Copy the service unit file:
   ```bash
   cp /root/projects/cg-gateway/systemd/cg-gateway.service /etc/systemd/system/
   ```

2. Reload systemd and start the service:
   ```bash
   systemctl daemon-reload
   systemctl enable cg-gateway
   systemctl start cg-gateway
   ```

3. Check service status and logs:
   ```bash
   systemctl status cg-gateway
   journalctl -u cg-gateway -f
   ```

---

## License

Internal proprietary research & integration gateway.
