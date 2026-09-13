import re
import json
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from app.core.tools import parse_tool_call_json, clean_malformed_json


class ParserState(str, Enum):
    TEXT = "TEXT"
    BUFFERING = "BUFFERING"
    IN_TOOL_CALL = "IN_TOOL_CALL"
    FINISH = "FINISH"


class LookaheadStreamParser:
    """
    Lookahead Streaming State Machine Parser for Tool Calls.
    
    States:
      - TEXT: default text emission
      - BUFFERING: candidate delimiter prefix match lookahead window
      - IN_TOOL_CALL: accumulating and streaming tool call JSON payload
      - FINISH: terminal state
      
    Guarantees:
      - O(1) amortized token streaming for delta.tool_calls without quadratic json.loads
      - Strict buffer limits: 64 KB per tool call, 1 MB per turn to prevent ReDoS / OOM
    """

    MAX_TOOL_CALL_BYTES = 64 * 1024       # 64 KB
    MAX_TURN_BYTES = 1024 * 1024          # 1 MB

    def __init__(
        self,
        start_delimiter: str = "<<<TOOL_CALL>>>",
        end_delimiter: str = "<<</TOOL_CALL>>>",
        max_tool_call_bytes: int = MAX_TOOL_CALL_BYTES,
        max_turn_bytes: int = MAX_TURN_BYTES
    ):
        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
        self.max_tool_call_bytes = max_tool_call_bytes
        self.max_turn_bytes = max_turn_bytes

        self.state = ParserState.TEXT
        self.text_buffer = ""
        self.tool_buffer = ""
        self.total_turn_bytes = 0

        self.current_tool_index = 0
        self.current_tool_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None
        self.tool_name_emitted = False
        self.streamed_arg_pos = 0

        # State tracking for argument streaming inside tool call
        self._arg_start_idx: Optional[int] = None
        self._arg_is_quoted = False
        self._arg_depth = 0
        self._in_quote = False
        self._escape_next = False
        self._scan_pos = 0

        self.tool_calls: List[Dict[str, Any]] = []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0 or self.current_tool_index > 0

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.has_tool_calls else "stop"

    def feed(self, chunk: str) -> List[Dict[str, Any]]:
        """
        Feeds a chunk of incoming text from upstream SSE and emits parsed delta events.
        Events:
          - {"type": "text", "content": str}
          - {"type": "tool_call_delta", "delta": {"tool_calls": [...]}}
        """
        if not chunk:
            return []

        chunk_len = len(chunk.encode("utf-8", errors="ignore"))
        self.total_turn_bytes += chunk_len
        if self.total_turn_bytes > self.max_turn_bytes:
            raise ValueError(f"Stream turn buffer exceeded maximum limit of {self.max_turn_bytes} bytes")

        events: List[Dict[str, Any]] = []
        pending = chunk

        while pending:
            if self.state in (ParserState.TEXT, ParserState.BUFFERING):
                self.text_buffer += pending
                pending = ""

                # Check if start delimiter is in text_buffer
                s_idx = self.text_buffer.find(self.start_delimiter)
                if s_idx != -1:
                    # Emit text prior to delimiter
                    prior_text = self.text_buffer[:s_idx]
                    if prior_text:
                        events.append({"type": "text", "content": prior_text})

                    # Everything after start_delimiter goes into tool_buffer
                    remainder = self.text_buffer[s_idx + len(self.start_delimiter):]
                    self.text_buffer = ""

                    # Transition to IN_TOOL_CALL
                    self._start_tool_call()
                    if remainder:
                        pending = remainder
                else:
                    # Check for partial prefix of start_delimiter at end of text_buffer
                    k = self._find_delimiter_prefix_suffix(self.text_buffer, self.start_delimiter)
                    if k > 0:
                        safe_text = self.text_buffer[:-k]
                        if safe_text:
                            events.append({"type": "text", "content": safe_text})
                        self.text_buffer = self.text_buffer[-k:]
                        self.state = ParserState.BUFFERING
                    else:
                        if self.text_buffer:
                            events.append({"type": "text", "content": self.text_buffer})
                        self.text_buffer = ""
                        self.state = ParserState.TEXT

            elif self.state == ParserState.IN_TOOL_CALL:
                self.tool_buffer += pending
                pending = ""

                if len(self.tool_buffer.encode("utf-8", errors="ignore")) > self.max_tool_call_bytes:
                    raise ValueError(f"Tool call buffer exceeded maximum limit of {self.max_tool_call_bytes} bytes")

                # Check if end delimiter is in tool_buffer
                e_idx = self.tool_buffer.find(self.end_delimiter)
                if e_idx != -1:
                    # Delimiter matched! Complete this tool call
                    payload = self.tool_buffer[:e_idx]
                    remainder = self.tool_buffer[e_idx + len(self.end_delimiter):]
                    self.tool_buffer = ""

                    final_events = self._finalize_tool_call(payload)
                    events.extend(final_events)

                    self.state = ParserState.TEXT
                    if remainder:
                        pending = remainder
                else:
                    # Check if end of tool_buffer contains partial prefix of end_delimiter
                    k = self._find_delimiter_prefix_suffix(self.tool_buffer, self.end_delimiter)
                    safe_len = len(self.tool_buffer) - k if k > 0 else len(self.tool_buffer)
                    stream_events = self._stream_tool_progress(safe_len)
                    events.extend(stream_events)

        return events

    def finish(self) -> List[Dict[str, Any]]:
        """
        Signals end of upstream stream, flushes remaining buffers, and finalizes state.
        """
        events: List[Dict[str, Any]] = []

        if self.state in (ParserState.TEXT, ParserState.BUFFERING):
            if self.text_buffer:
                events.append({"type": "text", "content": self.text_buffer})
                self.text_buffer = ""

        elif self.state == ParserState.IN_TOOL_CALL:
            # Unterminated tool call - attempt recovery
            if self.tool_buffer:
                final_events = self._finalize_tool_call(self.tool_buffer)
                events.extend(final_events)
                self.tool_buffer = ""

        self.state = ParserState.FINISH
        return events

    def _find_delimiter_prefix_suffix(self, text: str, delimiter: str) -> int:
        """
        Finds length of the longest suffix of `text` that is a non-empty proper prefix of `delimiter`.
        """
        max_k = min(len(text), len(delimiter) - 1)
        for k in range(max_k, 0, -1):
            if delimiter.startswith(text[-k:]):
                return k
        return 0

    def _start_tool_call(self) -> None:
        self.state = ParserState.IN_TOOL_CALL
        self.tool_buffer = ""
        self.current_tool_id = f"call_{uuid.uuid4().hex[:12]}"
        self.current_tool_name = None
        self.tool_name_emitted = False
        self.streamed_arg_pos = 0

        self._arg_start_idx = None
        self._arg_is_quoted = False
        self._arg_depth = 0
        self._in_quote = False
        self._escape_next = False
        self._scan_pos = 0
        self._arg_finished = False

    def _stream_tool_progress(self, safe_len: int) -> List[Dict[str, Any]]:
        """
        Extracts tool name and streams incremental argument fragments up to safe_len in O(1).
        """
        events: List[Dict[str, Any]] = []
        safe_text = self.tool_buffer[:safe_len]

        # 1. Extract and emit tool name if not yet emitted
        if not self.tool_name_emitted:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', safe_text)
            if name_match:
                self.current_tool_name = name_match.group(1)
                events.append({
                    "type": "tool_call_delta",
                    "delta": {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "id": self.current_tool_id,
                                "type": "function",
                                "function": {
                                    "name": self.current_tool_name,
                                    "arguments": ""
                                }
                            }
                        ]
                    }
                })
                self.tool_name_emitted = True

        if not self.tool_name_emitted:
            return events

        # 2. Locate arguments start position if not already identified
        if self._arg_start_idx is None:
            args_match = re.search(r'"arguments"\s*:\s*', safe_text)
            if args_match:
                idx = args_match.end()
                # Skip whitespace
                while idx < len(safe_text) and safe_text[idx].isspace():
                    idx += 1
                if idx < len(safe_text):
                    first_char = safe_text[idx]
                    self._arg_start_idx = idx
                    self._scan_pos = idx
                    if first_char == '"':
                        self._arg_is_quoted = True
                    elif first_char in ('{', '['):
                        self._arg_is_quoted = False
                        self._arg_depth = 0

        if self._arg_start_idx is None or self._arg_finished:
            return events

        # 3. Incrementally scan forward from self._scan_pos up to safe_len
        # to determine how many argument characters are valid and ready to stream
        arg_stream_end = self._scan_pos
        while self._scan_pos < len(safe_text):
            ch = safe_text[self._scan_pos]

            if self._arg_is_quoted:
                # Quoted string argument: "..."
                if self._escape_next:
                    self._escape_next = False
                    self._scan_pos += 1
                    arg_stream_end = self._scan_pos
                    continue
                if ch == '\\':
                    self._escape_next = True
                    self._scan_pos += 1
                    arg_stream_end = self._scan_pos
                    continue
                if ch == '"':
                    if self._scan_pos == self._arg_start_idx:
                        # Opening quote
                        self._scan_pos += 1
                        continue
                    else:
                        # Closing quote of string argument!
                        arg_stream_end = self._scan_pos
                        self._scan_pos += 1
                        self._arg_finished = True
                        break
                self._scan_pos += 1
                arg_stream_end = self._scan_pos

            else:
                # Object/array argument: {...} or [...]
                if self._escape_next:
                    self._escape_next = False
                    self._scan_pos += 1
                    arg_stream_end = self._scan_pos
                    continue
                if ch == '\\' and self._in_quote:
                    self._escape_next = True
                    self._scan_pos += 1
                    arg_stream_end = self._scan_pos
                    continue
                if ch == '"':
                    self._in_quote = not self._in_quote
                    self._scan_pos += 1
                    arg_stream_end = self._scan_pos
                    continue

                if not self._in_quote:
                    if ch in ('{', '['):
                        self._arg_depth += 1
                    elif ch in ('}', ']'):
                        self._arg_depth -= 1
                        if self._arg_depth == 0:
                            # Finished arguments object!
                            self._scan_pos += 1
                            arg_stream_end = self._scan_pos
                            self._arg_finished = True
                            break

                self._scan_pos += 1
                arg_stream_end = self._scan_pos

        # Extract characters between arg_start_idx and arg_stream_end
        if self._arg_is_quoted:
            raw_arg_chunk = safe_text[self._arg_start_idx + 1:arg_stream_end]
        else:
            raw_arg_chunk = safe_text[self._arg_start_idx:arg_stream_end]

        # Emit any new unstreamed characters
        if len(raw_arg_chunk) > self.streamed_arg_pos:
            new_chars = raw_arg_chunk[self.streamed_arg_pos:]
            events.append({
                "type": "tool_call_delta",
                "delta": {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {
                                "arguments": new_chars
                            }
                        }
                    ]
                }
            })
            self.streamed_arg_pos = len(raw_arg_chunk)

        return events

    def _finalize_tool_call(self, payload: str) -> List[Dict[str, Any]]:
        """
        Parses final tool call payload, flushes any remaining arguments, and records the tool call.
        """
        events: List[Dict[str, Any]] = []
        parsed = None
        try:
            parsed = parse_tool_call_json(payload)
        except Exception:
            # Fallback extraction
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', payload)
            t_name = name_match.group(1) if name_match else (self.current_tool_name or "unknown_function")
            args_match = re.search(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"[^"]*")', payload, flags=re.DOTALL)
            if args_match:
                t_args = clean_malformed_json(args_match.group(1))
            else:
                t_args = "{}"
            parsed = {"name": t_name, "arguments": t_args}

        tool_name = parsed["name"]
        final_args_str = parsed["arguments"]

        # If tool name wasn't emitted during streaming, emit it now
        if not self.tool_name_emitted:
            events.append({
                "type": "tool_call_delta",
                "delta": {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "id": self.current_tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": ""
                            }
                        }
                    ]
                }
            })
            self.tool_name_emitted = True

        # Emit any remaining argument characters
        if len(final_args_str) > self.streamed_arg_pos:
            remaining_args = final_args_str[self.streamed_arg_pos:]
            events.append({
                "type": "tool_call_delta",
                "delta": {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {
                                "arguments": remaining_args
                            }
                        }
                    ]
                }
            })
            self.streamed_arg_pos = len(final_args_str)

        # Record completed tool call
        self.tool_calls.append({
            "id": self.current_tool_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": final_args_str
            }
        })

        # Advance to next tool call index
        self.current_tool_index += 1
        return events
