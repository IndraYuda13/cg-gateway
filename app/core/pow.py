import base64
import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence, Optional, Dict

CORES = [8, 16, 24, 32]
DOCUMENT_KEYS = ["__reactContainer$fzelfjyxej8", "_reactListening5dehydibo78", "location"]
SCREEN_RESOLUTIONS = [[1920, 1080], [1440, 900], [2560, 1440], [3840, 2160]]
DEFAULT_POW_SCRIPT = "https://chatgpt.com/backend-api/sentinel/sdk.js"


def _legacy_parse_time() -> str:
    now = datetime.now(timezone(timedelta(hours=-5)))
    return now.strftime("%a %b %d %Y %H:%M:%S") + " GMT-0500 (Eastern Standard Time)"


def build_pow_config(
    user_agent: str,
    script_sources: Optional[Sequence[str]] = None,
    data_build: str = "",
) -> list[Any]:
    navigator_key = random.choice([
        "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
        "storage−[object StorageManager]",
        "locks−[object LockManager]",
        "appCodeName−Mozilla",
        "permissions−[object Permissions]",
        "share−function share() { [native code] }",
        "webdriver−false",
        "managed−[object NavigatorManagedData]",
        "canShare−function canShare() { [native code] }",
        "vendor−Google Inc.",
        "mediaDevices−[object MediaDevices]",
        "vibrate−function vibrate() { [native code] }",
        "storageBuckets−[object StorageBucketManager]",
        "mediaCapabilities−[object MediaCapabilities]",
        "cookieEnabled−true",
        "virtualKeyboard−[object VirtualKeyboard]",
        "product−Gecko",
        "presentation−[object Presentation]",
        "onLine−true",
        "mimeTypes−[object MimeTypeArray]",
        "credentials−[object CredentialsContainer]",
        "serviceWorker−[object ServiceWorkerContainer]",
        "keyboard−[object Keyboard]",
        "gpu−[object GPU]",
        "doNotTrack",
        "serial−[object Serial]",
        "pdfViewerEnabled−true",
        "language−en-US",
        "geolocation−[object Geolocation]",
        "userAgentData−[object NavigatorUAData]",
        "getUserMedia−function getUserMedia() { [native code] }",
        "sendBeacon−function sendBeacon() { [native code] }",
        "hardwareConcurrency−32",
        "windowControlsOverlay−[object WindowControlsOverlay]",
    ])
    window_key = random.choice([
        "0", "window", "self", "document", "name", "location", "customElements", "history",
        "navigation", "innerWidth", "innerHeight", "scrollX", "scrollY", "visualViewport",
        "screenX", "screenY", "outerWidth", "outerHeight", "devicePixelRatio", "screen",
        "chrome", "navigator", "onresize", "performance", "crypto", "indexedDB",
        "sessionStorage", "localStorage", "scheduler", "alert", "atob", "btoa", "fetch",
        "matchMedia", "postMessage", "queueMicrotask", "requestAnimationFrame", "setInterval",
        "setTimeout", "caches", "__NEXT_DATA__", "__BUILD_MANIFEST", "__NEXT_PRELOADREADY",
    ])
    script_source = random.choice(list(script_sources)) if script_sources else DEFAULT_POW_SCRIPT

    return [
        sum(random.choices(SCREEN_RESOLUTIONS, k=1)[0]),
        _legacy_parse_time(),
        4294705152,
        1,
        user_agent,
        script_source,
        data_build,
        "en-US",
        "en-US,es-US,en,es",
        random.random(),
        navigator_key,
        random.choice(DOCUMENT_KEYS),
        window_key,
        time.perf_counter() * 1000,
        str(uuid.uuid4()),
        "",
        random.choice(CORES),
        time.time() * 1000 - (time.perf_counter() * 1000),
        0, 0, 0, 0, 0, 0, 0,
    ]


def _pow_generate(
    seed: str,
    difficulty: str,
    config: list[Any],
    limit: int = 500000
) -> tuple[str, bool]:
    target = bytes.fromhex(difficulty)
    diff_len = len(difficulty) // 2
    seed_bytes = seed.encode()
    static_1 = (json.dumps(config[:3], separators=(",", ":"), ensure_ascii=False)[:-1] + ",").encode()
    static_2 = ("," + json.dumps(config[4:9], separators=(",", ":"), ensure_ascii=False)[1:-1] + ",").encode()
    static_3 = ("," + json.dumps(config[10:], separators=(",", ":"), ensure_ascii=False)[1:]).encode()

    for i in range(limit):
        final_json = static_1 + str(i).encode() + static_2 + str(i >> 1).encode() + static_3
        encoded = base64.b64encode(final_json)
        digest = hashlib.sha3_512(seed_bytes + encoded).digest()
        if digest[:diff_len] <= target:
            return encoded.decode(), True

    fallback = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + base64.b64encode(f'"{seed}"'.encode()).decode()
    return fallback, False


def build_legacy_requirements_token(
    user_agent: str,
    script_sources: Optional[Sequence[str]] = None,
    data_build: str = "",
) -> str:
    """
    Generates legacy P-token with 'gAAAAAC' prefix for Sentinel prepare request.
    """
    config = build_pow_config(user_agent, script_sources=script_sources, data_build=data_build)
    raw_json = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()
    return "gAAAAAC" + base64.b64encode(raw_json).decode()


def build_proof_token(
    seed: str,
    difficulty: str,
    user_agent: str,
    script_sources: Optional[Sequence[str]] = None,
    data_build: str = "",
) -> str:
    """
    Solves SHA3-512 Proof of Work challenge and formats token with 'gAAAAAB' prefix.
    """
    config = build_pow_config(user_agent, script_sources=script_sources, data_build=data_build)
    answer, solved = _pow_generate(seed, difficulty, config)
    if not solved:
        raise RuntimeError(f"Failed to solve proof token: difficulty={difficulty}")
    return "gAAAAAB" + answer


# ─────────────────────────────────────────────────────────────────────────────
# Turnstile Bytecode Virtual Machine Solver
# ─────────────────────────────────────────────────────────────────────────────

class OrderedMap:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.values: Dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        if key not in self.values:
            self.keys.append(key)
        self.values[key] = value


def _turnstile_to_str(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        special = {
            "window.Math": "[object Math]",
            "window.Reflect": "[object Reflect]",
            "window.performance": "[object Performance]",
            "window.localStorage": "[object Storage]",
            "window.Object": "function Object() { [native code] }",
            "window.Reflect.set": "function set() { [native code] }",
            "window.performance.now": "function () { [native code] }",
            "window.Object.create": "function create() { [native code] }",
            "window.Object.keys": "function keys() { [native code] }",
            "window.Math.random": "function random() { [native code] }",
        }
        return special.get(value, value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(value)
    return str(value)


def _xor_string(text: str, key: str) -> str:
    if not key:
        return text
    return "".join(chr(ord(ch) ^ ord(key[i % len(key)])) for i, ch in enumerate(text))


def solve_turnstile_token(dx: str, p: str) -> Optional[str]:
    """
    Deobfuscates and executes Turnstile bytecode VM using key p to generate
    the OpenAI-Sentinel-Turnstile-Token.
    """
    try:
        decoded = base64.b64decode(dx).decode()
        initial_tokens = json.loads(_xor_string(decoded, p))
    except Exception:
        return None

    process_map: Dict[Any, Any] = {}
    start_time = time.time()
    result = ""

    def func_1(e: float, t: float) -> None:
        process_map[e] = _xor_string(
            _turnstile_to_str(process_map.get(e)),
            _turnstile_to_str(process_map.get(t))
        )

    def func_2(e: float, t: Any) -> None:
        process_map[e] = t

    def func_3(e: str) -> None:
        nonlocal result
        result = base64.b64encode(e.encode()).decode()

    def func_5(e: float, t: float) -> None:
        current = process_map.get(e)
        incoming = process_map.get(t)
        if isinstance(current, (list, tuple)):
            process_map[e] = list(current) + [incoming]
            return
        if isinstance(current, (str, float)) or isinstance(incoming, (str, float)):
            process_map[e] = _turnstile_to_str(current) + _turnstile_to_str(incoming)
            return
        process_map[e] = "NaN"

    def func_6(e: float, t: float, n: float) -> None:
        tv = process_map.get(t)
        nv = process_map.get(n)
        if isinstance(tv, str) and isinstance(nv, str):
            value = f"{tv}.{nv}"
            process_map[e] = "https://chatgpt.com/" if value == "window.document.location" else value

    def func_7(e: float, *args: float) -> None:
        target = process_map.get(e)
        values = [process_map.get(arg) for arg in args]
        if isinstance(target, str) and target == "window.Reflect.set":
            if len(values) >= 3:
                obj, key_name, val = values[0], values[1], values[2]
                if hasattr(obj, "add"):
                    obj.add(str(key_name), val)
                elif isinstance(obj, dict):
                    obj[str(key_name)] = val
        elif callable(target):
            target(*values)

    def func_8(e: float, t: float) -> None:
        process_map[e] = process_map.get(t)

    def func_14(e: float, t: float) -> None:
        process_map[e] = json.loads(process_map.get(t))

    def json_default(o: Any) -> Any:
        if hasattr(o, "values") and isinstance(o.values, dict):
            return o.values
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    def func_15(e: float, t: float) -> None:
        process_map[e] = json.dumps(process_map.get(t), default=json_default)

    def func_17(e: float, t: float, *args: float) -> None:
        call_args = [process_map.get(arg) for arg in args]
        target = process_map.get(t)
        if target == "window.performance.now":
            elapsed_ns = time.time_ns() - int(start_time * 1e9)
            process_map[e] = (elapsed_ns + random.random()) / 1e6
        elif target == "window.Object.create":
            process_map[e] = OrderedMap()
        elif target == "window.Object.keys":
            if call_args and call_args[0] == "window.localStorage":
                process_map[e] = [
                    "STATSIG_LOCAL_STORAGE_INTERNAL_STORE_V4",
                    "STATSIG_LOCAL_STORAGE_STABLE_ID",
                    "client-correlated-secret",
                    "oai/apps/capExpiresAt",
                    "oai-did",
                    "STATSIG_LOCAL_STORAGE_LOGGING_REQUEST",
                    "UiState.isNavigationCollapsed.1",
                ]
            elif call_args and isinstance(call_args[0], OrderedMap):
                process_map[e] = call_args[0].keys
        elif target == "window.Math.random":
            process_map[e] = random.random()
        elif callable(target):
            process_map[e] = target(*call_args)

    def func_18(e: float) -> None:
        process_map[e] = base64.b64decode(_turnstile_to_str(process_map.get(e))).decode()

    def func_19(e: float) -> None:
        process_map[e] = base64.b64encode(_turnstile_to_str(process_map.get(e)).encode()).decode()

    def func_20(e: float, t: float, n: float, *args: float) -> None:
        if process_map.get(e) == process_map.get(t):
            target = process_map.get(n)
            if callable(target):
                target(*[process_map.get(arg) for arg in args])

    def func_21(*_: Any) -> None:
        return

    def func_23(e: float, t: float, *args: float) -> None:
        if process_map.get(e) is not None and callable(process_map.get(t)):
            process_map.get(t)(*args)

    def func_24(e: float, t: float, n: float) -> None:
        tv = process_map.get(t)
        nv = process_map.get(n)
        if isinstance(tv, str) and isinstance(nv, str):
            process_map[e] = f"{tv}.{nv}"

    process_map.update({
        1: func_1, 2: func_2, 3: func_3, 5: func_5, 6: func_6, 7: func_7, 8: func_8,
        9: initial_tokens, 10: "window", 14: func_14, 15: func_15, 16: p, 17: func_17,
        18: func_18, 19: func_19, 20: func_20, 21: func_21, 23: func_23, 24: func_24,
    })

    # Multi-stage VM execution: process tokens and detect dynamic expansion in process_map[9]
    cur_list = process_map[9]
    idx = 0
    while idx < len(cur_list):
        tok = cur_list[idx]
        try:
            fn = process_map.get(tok[0])
            if callable(fn):
                fn(*tok[1:])
        except Exception:
            pass

        # If stage 1 unpacked a stage 2 token list into process_map[9], switch execution
        if process_map[9] is not cur_list:
            cur_list = process_map[9]
            idx = 0
            continue
        idx += 1

    return result or None
