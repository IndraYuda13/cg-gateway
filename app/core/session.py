import time
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

from app.config import DATA_DIR, SESSION_FILE, MAX_SESSIONS


class SessionEntry:
    def __init__(
        self,
        session_id: Optional[str],
        conv_id: str,
        parent_message_id: Optional[str] = "client-created-root",
        created_at: Optional[float] = None,
        last_active: Optional[float] = None,
        turn_count: int = 1
    ):
        self.session_id = session_id
        self.conv_id = conv_id
        self.parent_message_id = parent_message_id or "client-created-root"
        self.created_at = created_at or time.time()
        self.last_active = last_active or time.time()
        self.turn_count = turn_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conv_id": self.conv_id,
            "parent_message_id": self.parent_message_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionEntry":
        return cls(
            session_id=d.get("session_id"),
            conv_id=d["conv_id"],
            parent_message_id=d.get("parent_message_id", "client-created-root"),
            created_at=d.get("created_at"),
            last_active=d.get("last_active"),
            turn_count=d.get("turn_count", 1)
        )


class SmartSessionPool:
    """
    Intelligent LRU Session Pool for ChatGPT Upstream.
    
    Features:
    1. Multi-tenant conversation isolation.
    2. Automatic continuity for multi-turn chats via Context Fingerprinting (SHA-256 of first user turn).
    3. Strict bounded size (MAX_SESSIONS) with automatic upstream deletion of LRU sessions.
    4. Auto-healing when conversation sessions expire or become invalid.
    """
    def __init__(self, file_path: Path = SESSION_FILE, max_size: int = MAX_SESSIONS):
        self.file_path = Path(file_path)
        self.max_size = max(1, max_size)
        self.pool: Dict[str, SessionEntry] = {}  # conv_id -> SessionEntry
        self.load()

    def load(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.pool[k] = SessionEntry.from_dict(v)
            except Exception as e:
                print(f"[SessionPool] Warning: Failed to load pool file: {e}")

    def save(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.file_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self.pool.items()}, f, indent=2)
            tmp_path.replace(self.file_path)
        except Exception as e:
            print(f"[SessionPool] Warning: Failed to save pool file: {e}")

    def compute_conv_id(
        self,
        messages: List[Dict[str, Any]],
        user: Optional[str] = None,
        explicit_session_id: Optional[str] = None
    ) -> str:
        """
        Derives an immutable conversation identifier:
        1. Explicit session ID passed in payload/header.
        2. Standard OpenAI `user` parameter.
        3. Fingerprint of the root user message (remains constant across multi-turn chats).
        """
        if explicit_session_id:
            return f"custom_{explicit_session_id.strip()}"
        if user and str(user).strip() not in ("", "none", "null"):
            return f"user_{str(user).strip()}"

        # Fingerprint from the first user prompt in the messages array
        first_user_content = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    first_user_content = c.strip()
                    break

        if first_user_content:
            h = hashlib.sha256(first_user_content.encode("utf-8")).hexdigest()[:16]
            return f"conv_{h}"

        return f"conv_anon_{int(time.time())}"

    def acquire(
        self,
        conv_id: str,
        create_fn: Optional[Callable[[], str]] = None,
        delete_fn: Optional[Callable[[str], bool]] = None,
        force_new: bool = False
    ) -> Tuple[Optional[str], str]:
        """
        Retrieves an active session for the conversation or initializes one.
        Evicts and deletes the oldest LRU session on upstream if pool is full.
        Returns: (session_id, parent_message_id)
        """
        now = time.time()

        if not force_new and conv_id in self.pool:
            entry = self.pool[conv_id]
            entry.last_active = now
            entry.turn_count += 1
            self.save()
            return entry.session_id, entry.parent_message_id or "client-created-root"

        # Evict LRU session if capacity reached
        if len(self.pool) >= self.max_size:
            lru_key = min(self.pool.keys(), key=lambda k: self.pool[k].last_active)
            old_entry = self.pool.pop(lru_key)
            print(f"[SessionPool] Pool full ({len(self.pool)+1}/{self.max_size}). Evicting LRU {lru_key} ({old_entry.session_id})")
            if delete_fn and old_entry.session_id:
                try:
                    delete_fn(old_entry.session_id)
                except Exception as e:
                    print(f"[SessionPool] Upstream delete error for {old_entry.session_id}: {e}")

        # Initialize fresh conversation entry
        self.pool[conv_id] = SessionEntry(
            session_id=None,
            conv_id=conv_id,
            parent_message_id="client-created-root",
            created_at=now,
            last_active=now,
            turn_count=1
        )
        self.save()
        return None, "client-created-root"

    def update_parent(self, conv_id: str, new_parent_id: Optional[str]) -> None:
        if conv_id in self.pool and new_parent_id:
            self.pool[conv_id].parent_message_id = new_parent_id
            self.pool[conv_id].last_active = time.time()
            self.save()

    def update_session_id(self, conv_id: str, session_id: str) -> None:
        if conv_id in self.pool and session_id:
            self.pool[conv_id].session_id = session_id
            self.pool[conv_id].last_active = time.time()
            self.save()

    def reset_conv(self, conv_id: str, delete_fn: Optional[Callable[[str], bool]] = None) -> None:
        if conv_id in self.pool:
            old = self.pool.pop(conv_id)
            self.save()
            if delete_fn and old.session_id:
                try:
                    delete_fn(old.session_id)
                except Exception:
                    pass


smart_pool = SmartSessionPool()
