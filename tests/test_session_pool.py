import tempfile
from pathlib import Path
from app.core.session import SmartSessionPool

def test_session_continuity():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_file = Path(tmpdir) / "test_pool.json"
        pool = SmartSessionPool(file_path=pool_file, max_size=5)

        # Turn 1: initial message without explicit session ID
        messages_1 = [{"role": "user", "content": "Kunci rahasia: ALPHA-998"}]
        conv_id_1 = pool.compute_conv_id(messages=messages_1, explicit_session_id=None)
        assert conv_id_1.startswith("conv_"), f"Unexpected conv_id_1: {conv_id_1}"

        sid_1, p_id_1 = pool.acquire(conv_id_1)
        assert sid_1 is None, f"Expected None, got {sid_1}"
        assert p_id_1 == "client-created-root"

        # Upstream response arrives for Turn 1
        upstream_session = "6aa3762d-9471-460d-83b5-71cb14b03657"
        upstream_msg_1 = "msg_turn_1_id"
        pool.update_session_id(conv_id_1, upstream_session)
        pool.update_parent(conv_id_1, upstream_msg_1)

        # Turn 2: client passes explicit_session_id matching upstream_session
        messages_2 = [{"role": "user", "content": "Apa kunci rahasia tadi?"}]
        conv_id_2 = pool.compute_conv_id(messages=messages_2, explicit_session_id=upstream_session)
        assert conv_id_2 == conv_id_1, f"Expected {conv_id_1}, got {conv_id_2}"

        sid_2, p_id_2 = pool.acquire(conv_id_2)
        assert sid_2 == upstream_session, f"Expected {upstream_session}, got {sid_2}"
        assert p_id_2 == upstream_msg_1, f"Expected {upstream_msg_1}, got {p_id_2}"

        # Upstream response arrives for Turn 2
        upstream_msg_2 = "msg_turn_2_id"
        pool.update_parent(conv_id_2, upstream_msg_2)

        # Turn 3: acquire directly using upstream_session string
        sid_3, p_id_3 = pool.acquire(upstream_session)
        assert sid_3 == upstream_session, f"Expected {upstream_session}, got {sid_3}"
        assert p_id_3 == upstream_msg_2, f"Expected {upstream_msg_2}, got {p_id_3}"

        # Turn 4: client passes custom_ prefix alias
        conv_id_4 = pool.compute_conv_id(messages=[], explicit_session_id=f"custom_{upstream_session}")
        assert conv_id_4 == conv_id_1, f"Expected {conv_id_1}, got {conv_id_4}"

        print("TEST PASS: SmartSessionPool continuity with explicit_session_id and upstream session_id verified!")


def test_update_session_id_indexes_both_original_and_final():
    """
    Verifies that update_session_id indexes both the original session_id (e.g. placeholder)
    and the final upstream conversation_id, ensuring multi-turn continuity regardless
    of which identifier the client sends in subsequent turns.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_file = Path(tmpdir) / "test_pool.json"
        pool = SmartSessionPool(file_path=pool_file, max_size=5)

        placeholder_sid = "placeholder-uuid-111"
        messages = [{"role": "user", "content": "Halo, nama saya Budi."}]
        conv_id = pool.compute_conv_id(messages=messages, explicit_session_id=placeholder_sid)
        assert conv_id == f"custom_{placeholder_sid}"

        # Turn 1: acquire fresh session
        sid_1, p_id_1 = pool.acquire(conv_id)
        assert p_id_1 == "client-created-root"

        # Upstream returns real conversation ID
        final_conv_id = "upstream-conv-uuid-222"
        pool.update_session_id(conv_id, final_conv_id, orig_session_id=placeholder_sid)
        pool.update_parent(conv_id, "msg-turn-1")

        # Verify resolve_key resolves BOTH original placeholder and final upstream ID
        assert pool.resolve_key(placeholder_sid) == conv_id
        assert pool.resolve_key(final_conv_id) == conv_id
        assert pool.resolve_key(f"custom_{placeholder_sid}") == conv_id
        assert pool.resolve_key(f"custom_{final_conv_id}") == conv_id

        # Verify compute_conv_id resolves to the same conversation for both IDs
        assert pool.compute_conv_id(messages=[], explicit_session_id=placeholder_sid) == conv_id
        assert pool.compute_conv_id(messages=[], explicit_session_id=final_conv_id) == conv_id

        # Verify acquire returns the active upstream conversation ID for both
        sid_from_placeholder, parent_from_placeholder = pool.acquire(placeholder_sid)
        assert sid_from_placeholder == final_conv_id
        assert parent_from_placeholder == "msg-turn-1"

        sid_from_final, parent_from_final = pool.acquire(final_conv_id)
        assert sid_from_final == final_conv_id
        assert parent_from_final == "msg-turn-1"

        # Turn 2: Upstream updates conversation ID again without explicit orig_session_id
        next_conv_id = "upstream-conv-uuid-333"
        pool.update_session_id(conv_id, next_conv_id)
        pool.update_parent(conv_id, "msg-turn-2")

        # All 3 identifiers must resolve to the same conversation
        assert pool.resolve_key(placeholder_sid) == conv_id
        assert pool.resolve_key(final_conv_id) == conv_id
        assert pool.resolve_key(next_conv_id) == conv_id

        # Verify persistence: reload pool from disk
        reloaded_pool = SmartSessionPool(file_path=pool_file, max_size=5)
        assert reloaded_pool.resolve_key(placeholder_sid) == conv_id
        assert reloaded_pool.resolve_key(final_conv_id) == conv_id
        assert reloaded_pool.resolve_key(next_conv_id) == conv_id
        assert reloaded_pool.compute_conv_id(messages=[], explicit_session_id=placeholder_sid) == conv_id

        sid_persisted, p_persisted = reloaded_pool.acquire(placeholder_sid)
        assert sid_persisted == next_conv_id
        assert p_persisted == "msg-turn-2"

        print("TEST PASS: update_session_id indexing of both original and final session IDs verified!")


if __name__ == "__main__":
    test_session_continuity()
    test_update_session_id_indexes_both_original_and_final()
