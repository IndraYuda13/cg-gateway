import re
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set

from app.api.schemas import (
    MessageItem,
    ContentPart,
    ImageUrlDetail,
    FileUrlDetail,
    ToolCall,
    ToolCallFunction,
    ResponsesRequest
)
from app.core.tools import sanitize_user_prompt


def extract_custom_tool_input(raw: Any) -> str:
    """
    Extracts the clean input string from a custom tool's argument payload.
    Supports JSON strings ('{"input": "..."}'), raw strings, or dicts.
    Handles partial streaming arguments gracefully.
    """
    if not raw:
        return ""
    if isinstance(raw, dict):
        val = raw.get("input", "")
        return val if isinstance(val, str) else json.dumps(val)
    if not isinstance(raw, str):
        return str(raw)

    s = raw.strip()
    # Try full json loads
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            if "input" in data:
                val = data["input"]
                return val if isinstance(val, str) else json.dumps(val)
            return json.dumps(data)
        if isinstance(data, str):
            return data
    except Exception:
        pass

    # Regex extraction for streaming partial "input": "..."
    m = re.search(r'"input"\s*:\s*"(.*)', s, flags=re.DOTALL)
    if m:
        val = m.group(1)
        # Strip unescaped trailing quote or trailing quote + closing brace
        end_m = re.search(r'(?<!\\)"\s*}?$', val)
        if end_m:
            val = val[:end_m.start()]
        try:
            val = json.loads(f'"{val}"')
        except Exception:
            val = val.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
        return val

    return s


def flatten_and_normalize_tools(
    tools: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Flattens namespace tools (e.g. type: "namespace", extracting inner tools),
    strips hosted server-side tools (web_search, image_generation),
    converts custom tools (e.g. exec, apply_patch) to OpenAI function schemas with:
      {"input": {"type": "string", "description": "Raw freeform input for this custom tool"}}
    and records them in freeform_tool_names.
    """
    if not tools or not isinstance(tools, list):
        return [], set()

    extracted_tools: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "namespace":
            inner_list = t.get("tools") or []
            if isinstance(inner_list, list):
                for inner in inner_list:
                    if isinstance(inner, dict):
                        extracted_tools.append(inner)
        else:
            extracted_tools.append(t)

    normalized: List[Dict[str, Any]] = []
    freeform_names: Set[str] = set()

    for t in extracted_tools:
        t_type = t.get("type")
        # Strip hosted server-side tools
        if t_type in ("web_search", "image_generation"):
            continue

        if t_type == "custom":
            name = t.get("name")
            if not name or not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            freeform_names.add(name)

            desc = t.get("description") or ""
            fmt = t.get("format")
            syntax = fmt.get("syntax") if isinstance(fmt, dict) else None
            definition = fmt.get("definition") if isinstance(fmt, dict) else None
            desc_parts = [p for p in [desc, syntax, definition] if p]
            combined_desc = "\n\n".join(desc_parts) if desc_parts else f"Custom tool: {name}"

            normalized.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": combined_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Raw freeform input for this custom tool"
                            }
                        },
                        "required": ["input"],
                        "additionalProperties": False
                    }
                }
            })
        elif t_type == "function" or "function" in t:
            fn_val = t.get("function")
            fn_obj = fn_val if isinstance(fn_val, dict) else t
            name = fn_obj.get("name", t.get("name", ""))
            if not name:
                continue
            normalized.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn_obj.get("description", ""),
                    "parameters": fn_obj.get("parameters", {"type": "object", "properties": {}})
                }
            })
        elif "name" in t:
            normalized.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                }
            })

    return normalized, freeform_names


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return str(content) if content else ""


def normalize_input_to_messages(
    input_items: Optional[List[Dict[str, Any]]],
    instructions: Optional[str] = None
) -> Tuple[List[MessageItem], Set[str]]:
    """
    Converts Codex `input` items and optional `instructions` to standard MessageItems:
      - role: "developer" | "system" -> system message (sanitized)
      - role: "user" -> user message (sanitized, with text/image/file support)
      - role: "assistant" -> assistant message (sanitized)
      - type: "function_call" -> assistant message with tool_calls
      - type: "custom_tool_call" -> assistant message with tool_calls (arguments: {"input": ...})
      - type: "function_call_output" | "custom_tool_call_output" -> role: "tool", tool_call_id, content (sanitized)
    Enforces delimiter sanitization on ALL input items and tool outputs.
    """
    messages: List[MessageItem] = []
    recorded_custom_tools: Set[str] = set()

    # Prepend instructions as system message if present
    if instructions and isinstance(instructions, str) and instructions.strip():
        sanitized_inst = sanitize_user_prompt(instructions.strip())
        messages.append(MessageItem(role="system", content=sanitized_inst))

    if not input_items:
        return messages, recorded_custom_tools

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        # Developer / System message
        if (item_type == "message" and role in ("developer", "system")) or (role in ("developer", "system") and item_type is None):
            raw_content = item.get("content", "")
            text = _extract_text_from_content(raw_content)
            if text:
                messages.append(MessageItem(role="system", content=sanitize_user_prompt(text)))

        # User message
        elif (item_type == "message" and role == "user") or (role == "user" and item_type is None):
            raw_content = item.get("content", "")
            if isinstance(raw_content, str):
                messages.append(MessageItem(role="user", content=sanitize_user_prompt(raw_content)))
            elif isinstance(raw_content, list):
                converted_parts: List[Any] = []
                for part in raw_content:
                    if isinstance(part, str):
                        converted_parts.append(ContentPart(type="text", text=sanitize_user_prompt(part)))
                    elif isinstance(part, dict):
                        p_type = part.get("type", "")
                        if p_type in ("input_text", "text"):
                            t = part.get("text", "")
                            converted_parts.append(ContentPart(type="text", text=sanitize_user_prompt(t)))
                        elif p_type in ("input_image", "image_url"):
                            img_val = part.get("image_url") or part.get("file_id") or ""
                            url_str = img_val.get("url", "") if isinstance(img_val, dict) else str(img_val)
                            detail = part.get("detail", "auto")
                            converted_parts.append(ContentPart(type="image_url", image_url=ImageUrlDetail(url=url_str, detail=detail)))
                        elif p_type in ("input_file", "file_url"):
                            file_val = part.get("file_url") or part.get("url") or ""
                            url_str = file_val.get("url", "") if isinstance(file_val, dict) else str(file_val)
                            converted_parts.append(ContentPart(
                                type="file_url",
                                file_url=FileUrlDetail(
                                    url=url_str,
                                    name=part.get("name"),
                                    mime_type=part.get("mime_type")
                                )
                            ))
                        else:
                            t = part.get("text", "")
                            if t:
                                converted_parts.append(ContentPart(type="text", text=sanitize_user_prompt(t)))
                messages.append(MessageItem(role="user", content=converted_parts))

        # Assistant message
        elif (item_type == "message" and role == "assistant") or (role == "assistant" and item_type is None):
            raw_content = item.get("content", "")
            text = _extract_text_from_content(raw_content)
            messages.append(MessageItem(role="assistant", content=sanitize_user_prompt(text)))

        # Standard function call
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            name = item.get("name", "")
            raw_args = item.get("arguments", {})
            if isinstance(raw_args, (dict, list)):
                args_str = json.dumps(raw_args)
            elif isinstance(raw_args, str):
                args_str = raw_args
            else:
                args_str = "{}"
            tc = ToolCall(id=call_id, type="function", function=ToolCallFunction(name=name, arguments=args_str))
            if messages and messages[-1].role == "assistant" and not messages[-1].content:
                if messages[-1].tool_calls is None:
                    messages[-1].tool_calls = []
                messages[-1].tool_calls.append(tc)
            else:
                messages.append(MessageItem(role="assistant", content="", tool_calls=[tc]))

        # Custom tool call
        elif item_type == "custom_tool_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            name = item.get("name", "")
            recorded_custom_tools.add(name)
            raw_input = item.get("input", "")
            if not isinstance(raw_input, str):
                raw_input = json.dumps(raw_input)
            args_str = json.dumps({"input": raw_input})
            tc = ToolCall(id=call_id, type="function", function=ToolCallFunction(name=name, arguments=args_str))
            if messages and messages[-1].role == "assistant" and not messages[-1].content:
                if messages[-1].tool_calls is None:
                    messages[-1].tool_calls = []
                messages[-1].tool_calls.append(tc)
            else:
                messages.append(MessageItem(role="assistant", content="", tool_calls=[tc]))

        # Function call output / Custom tool call output
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = item.get("call_id") or item.get("id") or ""
            raw_out = item.get("output", "")
            if not isinstance(raw_out, str):
                raw_out = json.dumps(raw_out)
            # Indirect delimiter sanitization on tool outputs
            sanitized_out = sanitize_user_prompt(raw_out)
            messages.append(MessageItem(role="tool", tool_call_id=call_id, content=sanitized_out))

    return messages, recorded_custom_tools


class ResponsesStreamAdapter:
    """
    Manages the lifecycle of OpenAI Responses API SSE events.
    Translates upstream ChatGPT tokens / tool call deltas into the Responses API sequence:
      - response.created
      - response.in_progress
      - reasoning:
          response.output_item.added
          response.reasoning_summary_part.added
          response.reasoning_summary_text.delta
          response.reasoning_summary_text.done
          response.output_item.done
      - message:
          response.output_item.added
          response.content_part.added
          response.output_text.delta
          response.output_text.done
          response.output_item.done
      - tool calls:
          standard:
            response.output_item.added (type: "function_call")
            response.function_call_arguments.delta
            response.function_call_arguments.done
            response.output_item.done
          custom:
            response.output_item.added (type: "custom_tool_call")
            response.custom_tool_call_input.delta
            response.custom_tool_call_input.done
            response.output_item.done
      - response.completed
    """

    def __init__(
        self,
        response_id: str,
        created: int,
        freeform_tool_names: Optional[Set[str]] = None
    ):
        self.response_id = response_id
        self.created = created
        self.freeform_tool_names = freeform_tool_names or set()
        self.seq = 1

        self.started = False
        self.completed_sent = False

        self.current_output_index = 0
        self.output_items: List[Dict[str, Any]] = []

        # Reasoning tracking
        self.reasoning_active = False
        self.reasoning_done = False
        self.reasoning_id: Optional[str] = None
        self.reasoning_index: Optional[int] = None
        self.reasoning_buf = ""

        # Message tracking
        self.msg_active = False
        self.msg_done = False
        self.msg_id: Optional[str] = None
        self.msg_index: Optional[int] = None
        self.msg_text_buf = ""

        # Tool calls tracking: dict mapping parser tool index to tool state
        self.tools: Dict[int, Dict[str, Any]] = {}

    def _make_event(self, event_type: str, data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        data["sequence_number"] = self.seq
        self.seq += 1
        return event_type, data

    def emit_initial(self) -> List[Tuple[str, Dict[str, Any]]]:
        if self.started:
            return []
        self.started = True
        evs = []
        evs.append(self._make_event("response.created", {
            "type": "response.created",
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created,
                "status": "in_progress",
                "background": False,
                "error": None,
                "output": []
            }
        }))
        evs.append(self._make_event("response.in_progress", {
            "type": "response.in_progress",
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created,
                "status": "in_progress"
            }
        }))
        return evs

    def handle_reasoning_delta(self, delta: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not delta:
            return []
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.extend(self.emit_initial())

        if not self.reasoning_active and not self.reasoning_done:
            self.reasoning_active = True
            self.reasoning_index = self.current_output_index
            self.current_output_index += 1
            self.reasoning_id = f"rs_{self.response_id}_{self.reasoning_index}"

            events.append(self._make_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": self.reasoning_index,
                "item": {
                    "id": self.reasoning_id,
                    "type": "reasoning",
                    "summary": []
                }
            }))
            events.append(self._make_event("response.reasoning_summary_part.added", {
                "type": "response.reasoning_summary_part.added",
                "item_id": self.reasoning_id,
                "output_index": self.reasoning_index,
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": ""
                }
            }))

        if self.reasoning_active:
            self.reasoning_buf += delta
            events.append(self._make_event("response.reasoning_summary_text.delta", {
                "type": "response.reasoning_summary_text.delta",
                "item_id": self.reasoning_id,
                "output_index": self.reasoning_index,
                "summary_index": 0,
                "delta": delta
            }))

        return events

    def finalize_reasoning(self) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        if self.reasoning_active and not self.reasoning_done:
            self.reasoning_active = False
            self.reasoning_done = True
            events.append(self._make_event("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done",
                "item_id": self.reasoning_id,
                "output_index": self.reasoning_index,
                "summary_index": 0,
                "text": self.reasoning_buf
            }))
            events.append(self._make_event("response.reasoning_summary_part.done", {
                "type": "response.reasoning_summary_part.done",
                "item_id": self.reasoning_id,
                "output_index": self.reasoning_index,
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": self.reasoning_buf
                }
            }))
            item = {
                "id": self.reasoning_id,
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": self.reasoning_buf
                    }
                ]
            }
            events.append(self._make_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": self.reasoning_index,
                "item": item
            }))
            self.output_items.append(item)
        return events

    def handle_text_delta(self, delta: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not delta:
            return []
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.extend(self.emit_initial())
        events.extend(self.finalize_reasoning())

        if not self.msg_active and not self.msg_done:
            self.msg_active = True
            self.msg_index = self.current_output_index
            self.current_output_index += 1
            self.msg_id = f"msg_{self.response_id}_{self.msg_index}"

            events.append(self._make_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": self.msg_index,
                "item": {
                    "id": self.msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            }))
            events.append(self._make_event("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": self.msg_id,
                "output_index": self.msg_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": ""
                }
            }))

        if self.msg_active:
            self.msg_text_buf += delta
            events.append(self._make_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": self.msg_id,
                "output_index": self.msg_index,
                "content_index": 0,
                "delta": delta,
                "logprobs": []
            }))

        return events

    def finalize_message(self) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        if self.msg_active and not self.msg_done:
            self.msg_active = False
            self.msg_done = True
            events.append(self._make_event("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": self.msg_id,
                "output_index": self.msg_index,
                "content_index": 0,
                "text": self.msg_text_buf,
                "logprobs": []
            }))
            events.append(self._make_event("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": self.msg_id,
                "output_index": self.msg_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": self.msg_text_buf
                }
            }))
            item = {
                "id": self.msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": self.msg_text_buf
                    }
                ]
            }
            events.append(self._make_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": self.msg_index,
                "item": item
            }))
            self.output_items.append(item)
        return events

    def handle_tool_call_delta(self, tc_delta: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.extend(self.emit_initial())
        events.extend(self.finalize_reasoning())
        events.extend(self.finalize_message())

        tc_index = tc_delta.get("index", 0)
        t_state = self.tools.get(tc_index)

        if not t_state:
            call_id = tc_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            fn_dict = tc_delta.get("function") or {}
            name = fn_dict.get("name") or ""
            is_custom = name in self.freeform_tool_names if name else False

            t_out_index = self.current_output_index
            self.current_output_index += 1

            t_state = {
                "index": tc_index,
                "call_id": call_id,
                "name": name,
                "is_custom": is_custom,
                "output_index": t_out_index,
                "raw_args": "",
                "emitted_input_len": 0,
                "added_sent": False,
                "done": False
            }
            self.tools[tc_index] = t_state

        # Update name if provided later
        fn_dict = tc_delta.get("function") or {}
        if fn_dict.get("name"):
            t_state["name"] = fn_dict["name"]
            t_state["is_custom"] = t_state["name"] in self.freeform_tool_names

        # Emit output_item.added as soon as name is available
        if not t_state["added_sent"] and t_state["name"]:
            t_state["added_sent"] = True
            is_cust = t_state["is_custom"]
            call_id = t_state["call_id"]
            name = t_state["name"]
            item_id = f"ctc_{call_id}" if is_cust else f"fc_{call_id}"
            t_state["item_id"] = item_id

            if is_cust:
                item = {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "input": ""
                }
            else:
                item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": ""
                }

            events.append(self._make_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": t_state["output_index"],
                "item": item
            }))

        # Handle arguments chunk
        new_chars = fn_dict.get("arguments", "")
        if new_chars:
            t_state["raw_args"] += new_chars
            item_id = t_state.get("item_id", f"call_{t_state['call_id']}")

            if t_state["is_custom"]:
                curr_input = extract_custom_tool_input(t_state["raw_args"])
                if len(curr_input) > t_state["emitted_input_len"]:
                    delta = curr_input[t_state["emitted_input_len"]:]
                    t_state["emitted_input_len"] = len(curr_input)
                    events.append(self._make_event("response.custom_tool_call_input.delta", {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": item_id,
                        "output_index": t_state["output_index"],
                        "delta": delta
                    }))
            else:
                events.append(self._make_event("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": t_state["output_index"],
                    "delta": new_chars
                }))

        return events

    def finalize_tool_call(self, tc_index: int, final_call: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        t_state = self.tools.get(tc_index)
        if not t_state or t_state["done"]:
            return events

        t_state["done"] = True
        is_cust = t_state["is_custom"]
        call_id = t_state["call_id"]
        name = t_state["name"] or (final_call.get("function", {}).get("name") if final_call else "unknown")
        item_id = t_state.get("item_id") or (f"ctc_{call_id}" if is_cust else f"fc_{call_id}")

        final_args = ""
        if final_call and "function" in final_call and "arguments" in final_call["function"]:
            final_args = final_call["function"]["arguments"]
        else:
            final_args = t_state["raw_args"]

        if is_cust:
            final_input = extract_custom_tool_input(final_args)
            if len(final_input) > t_state["emitted_input_len"]:
                rem = final_input[t_state["emitted_input_len"]:]
                t_state["emitted_input_len"] = len(final_input)
                events.append(self._make_event("response.custom_tool_call_input.delta", {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": item_id,
                    "output_index": t_state["output_index"],
                    "delta": rem
                }))
            events.append(self._make_event("response.custom_tool_call_input.done", {
                "type": "response.custom_tool_call_input.done",
                "item_id": item_id,
                "output_index": t_state["output_index"],
                "input": final_input
            }))
            item = {
                "id": item_id,
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "input": final_input
            }
        else:
            events.append(self._make_event("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": t_state["output_index"],
                "arguments": final_args
            }))
            item = {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": final_args
            }

        events.append(self._make_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": t_state["output_index"],
            "item": item
        }))
        self.output_items.append(item)
        return events

    def finalize_all_and_complete(self) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.extend(self.emit_initial())
        events.extend(self.finalize_reasoning())
        events.extend(self.finalize_message())

        # Finalize any uncompleted tool calls
        for idx in sorted(self.tools.keys()):
            if not self.tools[idx]["done"]:
                events.extend(self.finalize_tool_call(idx))

        if not self.completed_sent:
            self.completed_sent = True
            events.append(self._make_event("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created,
                    "status": "completed",
                    "background": False,
                    "error": None,
                    "output": self.output_items,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                }
            }))
        return events

    def emit_failed(self, error_message: str) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        events.extend(self.emit_initial())
        events.append(self._make_event("response.failed", {
            "type": "response.failed",
            "response": {
                "id": self.response_id,
                "status": "failed",
                "error": {
                    "message": error_message,
                    "type": "server_error",
                    "code": "responses_stream_error"
                }
            }
        }))
        return events


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """Formats an SSE message frame for the wire protocol."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
