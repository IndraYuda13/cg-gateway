import os
import json
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_FILE = DATA_DIR / "session_pool.json"
CREDENTIALS_FILE = DATA_DIR / "credentials.json"
ENV_FILE = BASE_DIR / ".env"

# Helper to load simple .env file if present
if ENV_FILE.exists():
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8560"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))

# Upstream Service Configuration
UPSTREAM_BASE_URL = os.getenv("CG_BASE_URL", "https://chatgpt.com")
FALLBACK_ACCOUNT_ID = "fb88ebc3-79ae-45ff-b8b1-2313efa099b4"
FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/152.0.0.0 Safari/537.36 Edg/152.0.0.0"
)

# Load dynamic credentials from credentials.json if present
_saved_creds = {}
if CREDENTIALS_FILE.exists():
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            _saved_creds = json.load(f)
    except Exception:
        _saved_creds = {}

ACCOUNT_ID = os.getenv(
    "CG_ACCOUNT_ID",
    os.getenv("ACCOUNT_ID", _saved_creds.get("account_id", FALLBACK_ACCOUNT_ID))
)
DEFAULT_TOKEN = os.getenv(
    "CG_TOKEN",
    os.getenv("DEFAULT_TOKEN", _saved_creds.get("token", ""))
)
DEFAULT_COOKIES = os.getenv(
    "CG_COOKIES",
    os.getenv("DEFAULT_COOKIES", _saved_creds.get("cookies", ""))
)
USER_AGENT = os.getenv(
    "USER_AGENT",
    _saved_creds.get("user_agent", FALLBACK_USER_AGENT)
)

# Supported Models
DEFAULT_MODELS = [
    {"id": "gpt-5-6", "owned_by": "openai"},
    {"id": "gpt-5-6-thinking", "owned_by": "openai"},
    {"id": "gpt-5-5", "owned_by": "openai"},
    {"id": "gpt-5-5-thinking", "owned_by": "openai"},
    {"id": "gpt-5-6-pro", "owned_by": "openai"},
    {"id": "gpt-6-pro", "owned_by": "openai"},
    {"id": "o3-pro", "owned_by": "openai"},
    {"id": "gpt-4o", "owned_by": "openai"},
    {"id": "gpt-4o-mini", "owned_by": "openai"},
]
