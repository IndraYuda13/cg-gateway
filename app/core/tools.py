import re
import json
import uuid
import secrets
from typing import List, Dict, Any, Optional, Tuple, Union

from app.api.schemas import ToolDefinition, FunctionDefinition


def generate_delimiters(nonce: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Generates a dynamic per-turn nonce and unique start/end delimiters.
    Returns: (nonce, start_delimiter, end_delimiter)
    """
    turn_nonce = nonce or secrets.token_hex(16)
    start_delim = f"<<<TOOL_CALL_{turn_nonce}>>>"
    end_delim = f"<<</TOOL_CALL_{turn_nonce}>>>"
    return turn_nonce, start_delim, end_delim



def sanitize_user_prompt(text: str) -> str:
    """
    Sanitizes user input to escape accidental or malicious lookalike delimiters,
    preventing prompt injection or delimiter spoofing.
    """
    if not text or not isinstance(text, str):
        return text or ""
    # Escape <<<TOOL_CALL and <<</TOOL_CALL patterns
    sanitized = re.sub(r"<<<(\s*/?\s*TOOL_CALL)", r"\<\<\<\1", text, flags=re.IGNORECASE)
    return sanitized


def format_tool_definitions(tools: List[Union[ToolDefinition, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Normalizes ToolDefinition or dict items into standard OpenAI tool dicts.
    """
    formatted: List[Dict[str, Any]] = []
    for t in tools:
        if isinstance(t, ToolDefinition):
            f_def = t.function
            if f_def:
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": f_def.name,
                        "description": f_def.description or "",
                        "parameters": f_def.parameters or {"type": "object", "properties": {}}
                    }
                })
        elif isinstance(t, dict):
            if t.get("type") == "function" or "function" in t:
                f_obj = t.get("function") or {}
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": f_obj.get("name", t.get("name", "")),
                        "description": f_obj.get("description", t.get("description", "")),
                        "parameters": f_obj.get("parameters", t.get("parameters", {"type": "object", "properties": {}}))
                    }
                })
            elif "name" in t:
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}})
                    }
                })
    return formatted


CODEX_BACKEND_EXECUTION_PROMPT = (
    "You are acting as the execution backend for OpenAI Codex running directly on the user's local machine.\n"
    "You HAVE direct access to the local environment and terminal via your attached tools (such as `exec`).\n"
    "When the user asks you to inspect the system, check VPS status/specs, explore files, run scripts, or perform tests:\n"
    "1. NEVER say that you lack access to the machine or terminal.\n"
    "2. NEVER ask the user to run commands manually when you have the tools to run them yourself.\n"
    "3. IMMEDIATELY call the appropriate tool (e.g. `exec`) using the required tool call delimiters to inspect the machine and return the actual output."
)


def compile_tool_prompt(
    tools: List[Union[ToolDefinition, Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    start_delimiter: str = "<<<TOOL_CALL>>>",
    end_delimiter: str = "<<</TOOL_CALL>>>",
    parallel_tool_calls: Optional[bool] = True,
    backend_contract: Optional[str] = None
) -> str:
    """
    Formats client JSON tool schemas into strict system prompt instructions.
    """
    formatted_tools = format_tool_definitions(tools)
    if not formatted_tools:
        if tools:
            raise ValueError("No valid function definitions found in tools parameter")
        return ""

    tools_json = json.dumps(formatted_tools, indent=2)

    tool_choice_instruction = ""
    if isinstance(tool_choice, str):
        tc_lower = tool_choice.strip().lower()
        if tc_lower == "none":
            return "Do NOT invoke any tools. Respond to the user using text only."
        elif tc_lower == "required":
            tool_choice_instruction = "You MUST call at least one tool to fulfill this request before giving your final answer."
        elif tc_lower == "auto":
            tool_choice_instruction = "You can choose to call one or more tools if needed, or respond directly with text."
        else:
            # Named function
            tool_choice_instruction = f"You MUST call the tool '{tool_choice}' to fulfill this request."
    elif isinstance(tool_choice, dict):
        fn_name = ""
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            fn_name = tool_choice["function"].get("name", "")
        elif "name" in tool_choice:
            fn_name = tool_choice["name"]
        if fn_name:
            tool_choice_instruction = f"You MUST call the tool '{fn_name}' to fulfill this request."

    parallel_note = (
        "You may invoke multiple tools in a single turn if needed."
        if parallel_tool_calls is not False
        else "Only invoke one tool at a time."
    )

    prompt = (
        "# TOOL CALLING INSTRUCTIONS\n\n"
        "You have access to the following external functions/tools:\n\n"
        f"```json\n{tools_json}\n```\n\n"
        "When you need to call a tool, you MUST wrap each tool call block in the following exact delimiters:\n"
        f"{start_delimiter}\n"
        '{"name": "<function_name>", "arguments": {<valid_json_arguments>}}\n'
        f"{end_delimiter}\n\n"
        "IMPORTANT RULES:\n"
        "1. The `arguments` key must be a valid JSON object matching the parameter schema.\n"
        f"2. {parallel_note}\n"
        "3. Do NOT make up or simulate the results of the tool. Output the tool call block and stop.\n"
        "4. The system will execute the function and provide the results in the next turn as a tool result message.\n"
    )
    if tool_choice_instruction:
        prompt += f"5. {tool_choice_instruction}\n"

    if backend_contract and backend_contract.strip():
        contract_text = backend_contract.strip()
        prompt = f"{contract_text}\n\n{prompt}\n\n{contract_text}"

    return prompt


def clean_malformed_json(raw: str) -> str:
    """
    Attempts to clean common minor JSON issues (trailing commas, unbalanced braces).
    """
    s = raw.strip()
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_tool_call_json(raw_payload: str) -> Dict[str, Any]:
    """
    Parses a tool call payload string from within the delimiters.
    Returns: {"name": str, "arguments": str} where arguments is a valid JSON string.
    Raises ValueError on invalid/unparseable payload.
    """
    cleaned = clean_malformed_json(raw_payload)
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: regex extraction of name and arguments
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
        if not name_match:
            raise ValueError(f"Failed to parse tool call JSON: could not find 'name' in '{raw_payload[:100]}'")
        tool_name = name_match.group(1)

        # Find arguments
        args_match = re.search(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"[^"]*")', cleaned, flags=re.DOTALL)
        if args_match:
            args_raw = args_match.group(1).strip()
            try:
                parsed_args = json.loads(clean_malformed_json(args_raw))
                if isinstance(parsed_args, str):
                    args_str = parsed_args
                else:
                    args_str = json.dumps(parsed_args)
            except Exception:
                args_str = args_raw
        else:
            args_str = "{}"
        return {"name": tool_name, "arguments": args_str}

    if not isinstance(data, dict):
        raise ValueError(f"Tool call payload must be a JSON object, got {type(data)}")

    tool_name = data.get("name")
    if not tool_name or not isinstance(tool_name, str):
        raise ValueError(f"Tool call missing valid 'name': {data}")

    args_val = data.get("arguments", {})
    if isinstance(args_val, (dict, list)):
        args_str = json.dumps(args_val)
    elif isinstance(args_val, str):
        # Validate or normalize
        try:
            parsed = json.loads(args_val)
            args_str = json.dumps(parsed) if isinstance(parsed, (dict, list)) else args_val
        except Exception:
            args_str = args_val
    else:
        args_str = "{}"

    return {"name": tool_name, "arguments": args_str}


def extract_tool_calls_from_text(
    text: str,
    start_delimiter: str,
    end_delimiter: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts all tool calls from text containing start_delimiter and end_delimiter.
    Returns: (cleaned_text_without_tool_blocks, list_of_tool_call_dicts)
    """
    if not text or start_delimiter not in text:
        return text, []

    tool_calls: List[Dict[str, Any]] = []
    cleaned_segments: List[str] = []
    idx = 0

    while True:
        s_pos = text.find(start_delimiter, idx)
        if s_pos == -1:
            cleaned_segments.append(text[idx:])
            break

        cleaned_segments.append(text[idx:s_pos])
        e_pos = text.find(end_delimiter, s_pos + len(start_delimiter))
        if e_pos == -1:
            # Unterminated delimiter - take the rest as payload
            raw_payload = text[s_pos + len(start_delimiter):].strip()
            idx = len(text)
        else:
            raw_payload = text[s_pos + len(start_delimiter):e_pos].strip()
            idx = e_pos + len(end_delimiter)

        if raw_payload:
            try:
                parsed = parse_tool_call_json(raw_payload)
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": parsed["arguments"]
                    }
                })
            except Exception as ex:
                # Malformed tool call
                cleaned_segments.append(f"\n[Malformed tool call: {ex}]\n")

    cleaned_text = "".join(cleaned_segments).strip()
    return cleaned_text, tool_calls
