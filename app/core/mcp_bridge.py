import os
import json
import socket
import ipaddress
import urllib.parse
import urllib.request
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from app.config import DATA_DIR
from app.api.schemas import ToolDefinition, FunctionDefinition


MCP_SERVERS_FILE = DATA_DIR / "mcp_servers.json"

SAFE_ENV_KEYS = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME", "USER",
    "PYTHONPATH", "NODE_PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "WINDIR"
}


class MCPSecurityError(ValueError):
    """Raised when an MCP operation violates the security sandbox."""
    pass


def validate_remote_url(url_str: str) -> str:
    """
    SSRF Protection: Validates remote URL, blocking loopback, private IP ranges,
    link-local addresses (e.g. AWS/Azure metadata 169.254.169.254), and internal hosts.
    """
    if not url_str:
        raise MCPSecurityError("Empty URL provided")

    parsed = urllib.parse.urlparse(url_str)
    if parsed.scheme.lower() not in ("http", "https"):
        raise MCPSecurityError(f"Disallowed URL scheme: {parsed.scheme} (must be http or https)")

    host = parsed.hostname
    if not host:
        raise MCPSecurityError("Missing hostname in URL")

    # Block well-known metadata / localhost domain strings
    host_lower = host.lower()
    if host_lower in ("localhost", "metadata.google.internal", "instance-data"):
        raise MCPSecurityError(f"Blocked internal hostname: {host}")

    # Check direct IP address
    try:
        ip_obj = ipaddress.ip_address(host)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            raise MCPSecurityError(f"SSRF protection: blocked private/internal IP {host}")
        return url_str
    except ValueError:
        pass

    # Resolve domain to IP and check
    try:
        resolved_ip = socket.gethostbyname(host)
        ip_obj = ipaddress.ip_address(resolved_ip)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            raise MCPSecurityError(
                f"SSRF protection: domain {host} resolved to blocked private/internal IP {resolved_ip}"
            )
    except socket.gaierror as e:
        raise MCPSecurityError(f"Failed to resolve hostname '{host}': {e}")

    return url_str


def build_sanitized_env(custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Builds a sanitized environment dictionary for local subprocess execution.
    Only whitelisted system variables and explicitly provided safe server env keys are included.
    """
    sanitized: Dict[str, str] = {}
    for k in SAFE_ENV_KEYS:
        if k in os.environ:
            sanitized[k] = os.environ[k]

    if custom_env and isinstance(custom_env, dict):
        for k, v in custom_env.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            # Filter out LD_PRELOAD or dangerous injections
            if k.upper().startswith("LD_") and k != "LD_LIBRARY_PATH":
                continue
            if k.upper() in ("DYLD_INSERT_LIBRARIES", "BASH_ENV", "ENV"):
                continue
            sanitized[k] = v

    return sanitized


class MCPBridge:
    """
    Lightweight Model Context Protocol (MCP) Client Bridge.
    Reads server definitions from data/mcp_servers.json, validates security sandbox,
    dispatches tools/list and tools/call to local (stdio) and remote (HTTP) MCP servers.
    """

    def __init__(self, config_file: Path = MCP_SERVERS_FILE):
        self.config_file = Path(config_file)
        self._servers: Dict[str, Dict[str, Any]] = {}
        self.load_servers()

    def load_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads server configurations from the JSON config file.
        """
        self._servers = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    servers = data.get("mcpServers") or data.get("servers") or {}
                    if isinstance(servers, dict):
                        self._servers = servers
            except Exception as e:
                print(f"[MCPBridge] Warning: Failed to load {self.config_file}: {e}")
        return self._servers

    def get_servers(self) -> Dict[str, Dict[str, Any]]:
        return self._servers

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp__")

    def split_mcp_tool_name(self, full_name: str) -> Tuple[str, str]:
        """
        Splits 'mcp__<server_name>__<tool_name>' into (server_name, tool_name).
        """
        if full_name.startswith("mcp__"):
            parts = full_name[5:].split("__", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        return "", full_name

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Queries all configured MCP servers and aggregates their tool definitions into OpenAI ToolDefinition dicts.
        """
        self.load_servers()
        all_tools: List[Dict[str, Any]] = []

        for s_name, s_conf in self._servers.items():
            try:
                tools = self._query_server_tools(s_name, s_conf)
                for t in tools:
                    t_name = t.get("name", "")
                    prefixed_name = f"mcp__{s_name}__{t_name}"
                    desc = t.get("description") or f"MCP tool from {s_name}"
                    schema = t.get("inputSchema") or t.get("parameters") or {"type": "object", "properties": {}}
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": desc,
                            "parameters": schema
                        }
                    })
            except Exception as e:
                print(f"[MCPBridge] Error querying tools from server '{s_name}': {e}")

        return all_tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches a tool call to the designated MCP server.
        Returns: Dict containing content/result or error.
        """
        s_name, raw_tool_name = self.split_mcp_tool_name(tool_name)
        if not s_name:
            # Fallback: check if tool_name matches a server with identical name
            if tool_name in self._servers:
                s_name = tool_name
                raw_tool_name = tool_name
            else:
                raise ValueError(f"Unknown MCP tool prefix: {tool_name}")

        if s_name not in self._servers:
            raise ValueError(f"Configured MCP server '{s_name}' not found")

        s_conf = self._servers[s_name]
        return self._execute_tool_call(s_name, s_conf, raw_tool_name, arguments)

    def _query_server_tools(self, s_name: str, conf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dispatches tools/list to a single MCP server.
        """
        # Case 1: Remote HTTP MCP Server
        if "url" in conf:
            url = validate_remote_url(conf["url"])
            headers = {"Content-Type": "application/json"}
            if "headers" in conf and isinstance(conf["headers"], dict):
                headers.update(conf["headers"])

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                result = res_data.get("result", {})
                return result.get("tools", [])

        # Case 2: Local Subprocess stdio MCP Server
        elif "command" in conf:
            cmd = conf["command"]
            args = conf.get("args", [])
            env = build_sanitized_env(conf.get("env"))

            # Security sandbox: strictly shell=False, command must not contain shell metacharacters
            if any(char in str(cmd) for char in (";", "&", "|", "`", "$", ">", "<")):
                raise MCPSecurityError(f"Disallowed metacharacters in MCP command: {cmd}")

            full_cmd = [cmd] + [str(a) for a in args]
            proc = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                env=env
            )
            req_line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
            try:
                stdout, _ = proc.communicate(input=req_line, timeout=15)
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                for l in lines:
                    try:
                        data = json.loads(l)
                        if "result" in data and "tools" in data["result"]:
                            return data["result"]["tools"]
                    except Exception:
                        continue
            finally:
                if proc.poll() is None:
                    proc.kill()

        return []

    def _execute_tool_call(
        self,
        s_name: str,
        conf: Dict[str, Any],
        raw_tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dispatches tools/call to a single MCP server.
        """
        # Case 1: Remote HTTP MCP Server
        if "url" in conf:
            url = validate_remote_url(conf["url"])
            headers = {"Content-Type": "application/json"}
            if "headers" in conf and isinstance(conf["headers"], dict):
                headers.update(conf["headers"])

            payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": raw_tool_name,
                    "arguments": arguments
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data.get("result", res_data)

        # Case 2: Local Subprocess stdio MCP Server
        elif "command" in conf:
            cmd = conf["command"]
            args = conf.get("args", [])
            env = build_sanitized_env(conf.get("env"))

            if any(char in str(cmd) for char in (";", "&", "|", "`", "$", ">", "<")):
                raise MCPSecurityError(f"Disallowed metacharacters in MCP command: {cmd}")

            full_cmd = [cmd] + [str(a) for a in args]
            proc = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                env=env
            )
            req_line = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": raw_tool_name,
                    "arguments": arguments
                }
            }) + "\n"
            try:
                stdout, stderr = proc.communicate(input=req_line, timeout=30)
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                for l in reversed(lines):
                    try:
                        data = json.loads(l)
                        if "result" in data:
                            return data["result"]
                        if "error" in data:
                            return data
                    except Exception:
                        continue
                return {"content": [{"type": "text", "text": stdout.strip()}], "raw_stdout": stdout}
            finally:
                if proc.poll() is None:
                    proc.kill()

        raise ValueError(f"Server '{s_name}' does not specify a valid url or command")


mcp_bridge = MCPBridge()
