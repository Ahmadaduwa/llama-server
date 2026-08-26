# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "httpx",
#     "uvicorn",
#     "python-dotenv",
# ]
# ///
# Runs on the gateway machine (100.119.233.96).
#
# ── Features ───────────────────────────────────────────────────────────────────────────────────
# 1. Non-blocking Async Gateway:
#    - Forwards with httpx.AsyncClient without timeout (HTTP_TIMEOUT = None) for unlimited generation.
#    - Health, models, tags, and status endpoints answer instantly even during long generations.
#    - Client disconnects raise CancelledError which cancels upstream llama-server generation.
# 2. Shared Drive Logging (Z:\) with Network Disconnect Tolerance:
#    - Writes to primary log on shared drive (Z:\server.log) and local backup (server.log).
#    - Tolerates network share disconnects/lag without crashing or blocking the server.
#    - Auto-reconnects and resumes writing to shared drive when it comes back online.
# 3. Inter-Machine Communication & Status:
#    - Real-time gateway status JSON (Z:\gateway_status.json & gateway_status.json) with heartbeat.
#    - Backend llama-server output mirrored to Z:\llama-server.log.
# 4. On-demand model tier loading/unloading, VRAM auto-downgrade, and Ollama compatibility.

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

# Ensure standard output streams support UTF-8 on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from uvicorn.config import Config, LOGGING_CONFIG
from uvicorn.server import Server

# ── Base Directory & Environment ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("API_KEY", "")
UPSTREAM = "http://localhost:8080"  # Backing llama-server.exe

# ── Shared Drive & Local Paths Configuration ───────────────────────────────────────────────────
def _normalize_dir(d: str) -> str:
    if not d:
        return ""
    d = d.strip().replace("/", "\\")
    if d.endswith(":") and len(d) == 2:
        d += "\\"
    return d

LOG_DIR = _normalize_dir(os.getenv("LOG_DIR", "Z:\\"))


def _resolve_path(env_var: str, default_filename: str, fallback_dir: str = "") -> str:
    val = os.getenv(env_var)
    if val:
        return val.strip()
    if fallback_dir:
        return os.path.join(fallback_dir, default_filename)
    return os.path.join(BASE_DIR, default_filename)


LOG_FILE = _resolve_path("LOG_FILE", "server.log", LOG_DIR)
STATUS_FILE = _resolve_path("STATUS_FILE", "gateway_status.json", LOG_DIR)
LLAMA_LOG_FILE = _resolve_path("LLAMA_LOG_FILE", "llama-server.log", LOG_DIR)

LOCAL_LOG_FILE = os.getenv("LOCAL_LOG_FILE", os.path.join(BASE_DIR, "server.log"))
LOCAL_STATUS_FILE = os.getenv("LOCAL_STATUS_FILE", os.path.join(BASE_DIR, "gateway_status.json"))
LOCAL_LLAMA_LOG_FILE = os.getenv("LOCAL_LLAMA_LOG_FILE", os.path.join(BASE_DIR, "llama-server.log"))


# ── Resilient Dual Logger (Shared Drive + Local Fallback) ──────────────────────────────────────
class SafeDualLogger:
    """
    Thread-safe logger that writes log lines to:
    1. Local fallback file (server.log) -> Guaranteed local record, never lost.
    2. Primary shared drive file (e.g. Z:\\server.log) -> For inter-machine monitoring.

    If the shared drive (Z:\\) is disconnected or unreachable:
    - Never crashes or hangs the async server.
    - Emits a throttled warning to console and local log.
    - Keeps recording locally.
    - Automatically resumes writing to the shared drive upon reconnection.
    """
    def __init__(self, primary_path: str | None, fallback_path: str):
        self.primary_path = primary_path
        self.fallback_path = fallback_path
        self._lock = threading.Lock()
        self._primary_online = True
        self._last_warn_time = 0.0

    @property
    def is_primary_online(self) -> bool:
        return self._primary_online

    def write(self, message: str):
        line = message if message.endswith("\n") else f"{message}\n"

        with self._lock:
            # 1. Local fallback log (local disk is always safe)
            try:
                if self.fallback_path:
                    f_dir = os.path.dirname(self.fallback_path)
                    if f_dir and not os.path.exists(f_dir):
                        os.makedirs(f_dir, exist_ok=True)
                    with open(self.fallback_path, "a", encoding="utf-8", errors="replace") as f:
                        f.write(line)
            except Exception:
                pass

            # 2. Shared drive log (Z:\server.log)
            if self.primary_path:
                try:
                    p_dir = os.path.dirname(self.primary_path)
                    if p_dir and not os.path.exists(p_dir):
                        os.makedirs(p_dir, exist_ok=True)
                    with open(self.primary_path, "a", encoding="utf-8", errors="replace") as f:
                        f.write(line)

                    if not self._primary_online:
                        self._primary_online = True
                        now_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                        recon_msg = f"{now_str} 🟢 [Shared Drive] Reconnected to shared log: {self.primary_path}\n"
                        try:
                            print(recon_msg.strip())
                        except Exception:
                            pass
                        try:
                            with open(self.primary_path, "a", encoding="utf-8", errors="replace") as f:
                                f.write(recon_msg)
                        except Exception:
                            pass
                except Exception as e:
                    now = time.time()
                    if self._primary_online or (now - self._last_warn_time > 60):
                        self._primary_online = False
                        self._last_warn_time = now
                        now_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                        warn_msg = f"{now_str} ⚠️ [Shared Drive] Cannot write to '{self.primary_path}': {e}. Logging to local '{self.fallback_path}' only."
                        try:
                            print(warn_msg)
                        except Exception:
                            pass


dual_logger = SafeDualLogger(primary_path=LOG_FILE, fallback_path=LOCAL_LOG_FILE)


def ts_print(*args, **kwargs):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    msg = " ".join(str(a) for a in args)
    print(f"{now} {msg}", **kwargs)
    dual_logger.write(f"{now} {msg}")


class SafeDualLogHandler(logging.Handler):
    """Logging handler to route Uvicorn logs to SafeDualLogger."""
    def __init__(self, target_logger: SafeDualLogger):
        super().__init__()
        self.target_logger = target_logger

    def emit(self, record):
        try:
            msg = self.format(record)
            self.target_logger.write(msg)
        except Exception:
            self.handleError(record)


# ── Machine-to-Machine Status Sharing (JSON) ──────────────────────────────────────────────────
def safe_write_json(file_path: str, data: dict):
    if not file_path:
        return
    try:
        p_dir = os.path.dirname(file_path)
        if p_dir and not os.path.exists(p_dir):
            os.makedirs(p_dir, exist_ok=True)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with open(file_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
    except Exception:
        pass


def update_status(status: str, extra: dict | None = None):
    curr_model = None
    act_reqs = 0
    if "manager" in globals() and manager:
        curr_model = manager.current_model
        act_reqs = manager.active_requests

    data = {
        "status": status,
        "current_model": curr_model,
        "active_requests": act_reqs,
        "last_heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_epoch": time.time(),
        "gateway_url": "http://100.119.233.96:11434",
        "shared_drive_connected": dual_logger.is_primary_online,
    }
    if extra:
        data.update(extra)

    safe_write_json(LOCAL_STATUS_FILE, data)
    if STATUS_FILE:
        safe_write_json(STATUS_FILE, data)


# ── Uvicorn Logging Configuration ─────────────────────────────────────────────────────────────
log_config = LOGGING_CONFIG.copy()
log_config["formatters"]["access"]["fmt"] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
log_config["formatters"]["default"]["fmt"] = '[%(asctime)s] %(levelprefix)s %(message)s'
log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


# ── FastAPI App & HTTP Client Setup ───────────────────────────────────────────────────────────
app = FastAPI(title="Pentest LLM Gateway (async, non-blocking)")

# Disabled HTTP timeout: None allows indefinite streaming and large reasoning context generations
# without premature aborts.
HTTP_TIMEOUT = None
http_client: httpx.AsyncClient | None = None


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    safe_paths = [
        "/api/tags", "/v1/models", "/api/v1/models", "/api/version", "/version",
        "/v1/props", "/props", "/api/show", "/health", "/slots", "/metrics",
    ]
    if request.url.path in safe_paths:
        return await call_next(request)

    if API_KEY:
        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"error": "Unauthorized: Invalid or missing API Key"})

    return await call_next(request)


# ── Smart LLM Manager (Lifecycle, VRAM Auto-downgrade & Subprocess) ────────────────────────────
class SmartLLMManager:
    def __init__(self):
        self.current_process = None
        self.current_model = None
        self.last_active_time = time.time()
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self._llama_log_handle = None

        # 🟢 Adjust your folder path here
        base_dir = "D:\\Program\\llama-b10054-bin-win-cuda-13.3-x64"
        exe = os.path.join(base_dir, "llama-server.exe")

        model_9b = os.path.join(base_dir, "model\\Qwen3.5-9B-GGUF\\Qwen3.5-9B-Q4_K_M.gguf")
        model_9b_uncen = os.path.join(base_dir, "model\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
        model_ornith = os.path.join(base_dir, "model\\Ornith-1.5-9B-uncensored\\Ornith-1.5-9B-uncensored.Q4_K_M.gguf")
        model_ornith_i1 = os.path.join(base_dir, "model\\Ornith-1.5-9B-uncensored-i1-GGUF\\Ornith-1.5-9B-uncensored.i1-Q5_K_M.gguf")
        model_27b_cyber = os.path.join(base_dir, "model\\Qwen3.8-27B-Uncensored-Cyber-i1-GGUF\\Qwen3.8-27B-Uncensored-Cyber.i1-IQ4_XS.gguf")

        model_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\Qwen3.5-4B-Q4_K_M.gguf")
        mmproj_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\mmproj-Qwen3.5-4B-BF16.gguf")

        # 🟢 Tier command repository (llama-server runs in the background on port 8080).
        # Prefix with & so PowerShell understands commands with quotes.
        self.configs = {
            # Qwen 3.8 27B Uncensored Cyber (IQ4_XS) - 65K Context Budget, KV Cache 4-bit, 26 Layers GPU Offload, 8-10 CPU Threads, MTP Speculative Decoding (Max Draft 2)
            "Qwen3.8-27B-65k-cyber": f'& "{exe}" -m "{model_27b_cyber}" --port 8080 -ngl 26 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja --spec-type draft-mtp --spec-draft-n-max 2',
            "Qwen3.8-27B-32k-cyber": f'& "{exe}" -m "{model_27b_cyber}" --port 8080 -ngl 26 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja --spec-type draft-mtp --spec-draft-n-max 2',
            "Qwen3.8-27B-16k-cyber": f'& "{exe}" -m "{model_27b_cyber}" --port 8080 -ngl 26 -c 16384 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja --spec-type draft-mtp --spec-draft-n-max 2',

            # Ornith 1.5 9B Tiers
            "Ornith-1.5-9B-32k": f'& "{exe}" -m "{model_ornith}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-65k": f'& "{exe}" -m "{model_ornith}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-i1-32k": f'& "{exe}" -m "{model_ornith_i1}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-i1-65k": f'& "{exe}" -m "{model_ornith_i1}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',

            # Qwen 3.5 9B Uncensored Tiers
            "Qwen3.5-9B-32k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-65k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-132k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 131072 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-192k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 196608 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',

            # Qwen 3.5 9B Standard Tiers
            "Qwen3.5-9B-32k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-65k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-132k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 131072 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-192k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 196608 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',

            # Qwen 3.5 4B Vision/Base Tier
            "Qwen3.5-4B-64k": f'& "{exe}" -m "{model_4b}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 10 --parallel 1 --cont-batching --jinja',
        }

    LOAD_TIMEOUT_SECONDS = int(os.getenv("GATEWAY_LOAD_TIMEOUT", "180"))
    IDLE_UNLOAD_SECONDS = int(os.getenv("GATEWAY_IDLE_UNLOAD_SECONDS", "600"))

    async def _server_healthy(self) -> bool:
        """Quick async liveness probe of the backing llama-server on port 8080."""
        try:
            r = await http_client.get(f"{UPSTREAM}/health", timeout=2.0)
            return r.status_code == 200
        except (httpx.HTTPError, Exception):
            return False

    def _downgrade_tier(self, model_name: str):
        """The next-smaller-context tier of the SAME model family on VRAM pressure."""
        m = re.match(r"^(.*?)-(\d+)k(.*)$", model_name or "")
        if not m:
            return None
        prefix, ctx, suffix = m.group(1), int(m.group(2)), (m.group(3) or "")
        smaller = sorted(
            (int(re.match(rf"^{re.escape(prefix)}-(\d+)k", k).group(1))
             for k in self.configs
             if re.match(rf"^{re.escape(prefix)}-\d+k{re.escape(suffix)}$", k)),
            reverse=True,
        )
        nxt = next((c for c in smaller if c < ctx), None)
        return f"{prefix}-{nxt}k{suffix}" if nxt is not None else None

    async def ensure_model(self, model_name):
        async with self.lock:
            clean_name = (model_name or "").replace("openai/", "").replace(":latest", "").strip()
            
            # Map friendly aliases
            aliases = {
                "qwen3.8-27b": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-65k": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-70k": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-cyber": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-uncensored-cyber": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-uncensored-cyber.i1-iq4_xs": "Qwen3.8-27B-65k-cyber",
                "qwen3.8-27b-iq4_xs": "Qwen3.8-27B-65k-cyber",
                "qwen-27b": "Qwen3.8-27B-65k-cyber",
                "27b": "Qwen3.8-27B-65k-cyber",
            }
            if clean_name.lower() in aliases:
                clean_name = aliases[clean_name.lower()]

            if clean_name in self.configs:
                model_name = clean_name
            else:
                case_map = {k.lower(): k for k in self.configs}
                if clean_name.lower() in case_map:
                    model_name = case_map[clean_name.lower()]
                else:
                    ts_print(f"⚠️ Model '{model_name}' not in known tier list -> defaulting to 'Qwen3.8-27B-65k-cyber' if requested 27B or 'Ornith-1.5-9B-65k'")
                    if "27b" in clean_name.lower():
                        model_name = "Qwen3.8-27B-65k-cyber"
                    else:
                        model_name = "Ornith-1.5-9B-65k"

            if self.current_model == model_name and self.current_process is not None:
                proc_dead = self.current_process.poll() is not None
                if proc_dead or not await self._server_healthy():
                    ts_print(f"⚠️ [Smart Gateway] Backing llama-server not responding (proc_dead={proc_dead}) -> reloading [{model_name}]...")
                    self.unload()
                else:
                    self.last_active_time = time.time()
                    return

            self.unload()
            while True:
                ts_print(f"\n🚀 [Smart Gateway] Loading [{model_name}] into VRAM...")
                update_status("loading", {"loading_model": model_name})
                await asyncio.sleep(2)
                cmd = self.configs[model_name]

                # Open local log file for llama-server process output
                try:
                    if self._llama_log_handle and not self._llama_log_handle.closed:
                        self._llama_log_handle.close()
                except Exception:
                    pass

                try:
                    self._llama_log_handle = open(LOCAL_LLAMA_LOG_FILE, "a", encoding="utf-8", errors="replace", buffering=1)
                    stdout_target = self._llama_log_handle
                except Exception:
                    stdout_target = subprocess.DEVNULL

                self.current_process = subprocess.Popen(
                    ["powershell", "-Command", cmd],
                    stdout=stdout_target,
                    stderr=subprocess.STDOUT,
                )
                self.current_model = model_name
                self.last_active_time = time.time()

                deadline = time.time() + self.LOAD_TIMEOUT_SECONDS
                ready = False
                while time.time() < deadline:
                    if self.current_process.poll() is not None:
                        break  # child exited (OOM / error) → downgrade
                    if await self._server_healthy():
                        ready = True
                        break
                    await asyncio.sleep(1)

                if ready:
                    ts_print(f"✅ [Smart Gateway] Successfully loaded [{model_name}]! Ready to forward.")
                    update_status("ready", {"current_model": model_name})
                    break

                self.unload()
                smaller = self._downgrade_tier(model_name)
                if smaller and smaller in self.configs:
                    ts_print(f"⚠️ [Smart Gateway] [{model_name}] failed to load within {self.LOAD_TIMEOUT_SECONDS}s (likely VRAM OOM). Auto-downgrading → [{smaller}].")
                    model_name = smaller
                    continue

                ts_print(f"❌ [Smart Gateway] [{model_name}] failed to load and no smaller tier available.")
                self.current_model = None
                update_status("error", {"error": f"Failed to load {model_name}"})
                break

    def unload(self):
        if self.current_process or self._llama_log_handle:
            ts_print("💤 [Smart Gateway] Clearing GPU VRAM (Force Kill llama-server)...")
            subprocess.run(
                ["powershell", "-Command", "Stop-Process -Name 'llama-server' -Force -ErrorAction SilentlyContinue"],
                check=False,
            )
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=1)
                except Exception:
                    try:
                        self.current_process.kill()
                    except Exception:
                        pass
                self.current_process = None

            if self._llama_log_handle:
                try:
                    self._llama_log_handle.close()
                except Exception:
                    pass
                self._llama_log_handle = None

            self.current_model = None
            update_status("idle", {"current_model": None})

    async def auto_unload_task(self):
        idle = self.IDLE_UNLOAD_SECONDS
        mins = idle // 60
        while True:
            await asyncio.sleep(30)
            if self.current_process and self.active_requests == 0 and (time.time() - self.last_active_time > idle):
                async with self.lock:
                    if self.current_process and self.active_requests == 0 and (time.time() - self.last_active_time > idle):
                        ts_print(f"⏳ [Smart Gateway] Idle > {mins} min -> unloading model to save VRAM")
                        self.unload()


manager = SmartLLMManager()


# ── Background Tasks (Heartbeat & Log Mirroring) ──────────────────────────────────────────────
async def heartbeat_status_task():
    """Periodically writes status JSON so other machines can verify connectivity & load state."""
    while True:
        await asyncio.sleep(10)
        try:
            status_str = "busy" if manager.active_requests > 0 else ("ready" if manager.current_model else "idle")
            update_status(status_str)
        except Exception:
            pass


async def sync_llama_log_task():
    """Safely mirrors newly appended bytes from local llama-server.log to shared drive Z:\llama-server.log."""
    read_pos = 0
    if os.path.exists(LOCAL_LLAMA_LOG_FILE):
        try:
            read_pos = os.path.getsize(LOCAL_LLAMA_LOG_FILE)
        except Exception:
            read_pos = 0

    while True:
        await asyncio.sleep(2)
        if not LLAMA_LOG_FILE or not os.path.exists(LOCAL_LLAMA_LOG_FILE):
            continue
        try:
            current_size = os.path.getsize(LOCAL_LLAMA_LOG_FILE)
            if current_size > read_pos:
                with open(LOCAL_LLAMA_LOG_FILE, "r", encoding="utf-8", errors="replace") as lf:
                    lf.seek(read_pos)
                    new_data = lf.read()
                    read_pos = lf.tell()

                if new_data:
                    l_dir = os.path.dirname(LLAMA_LOG_FILE)
                    if l_dir and not os.path.exists(l_dir):
                        os.makedirs(l_dir, exist_ok=True)
                    with open(LLAMA_LOG_FILE, "a", encoding="utf-8", errors="replace") as zf:
                        zf.write(new_data)
            elif current_size < read_pos:
                read_pos = 0
        except Exception:
            # Network share may be temporarily down; retry next cycle without failing
            pass


# ── Lifespan Events ────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global http_client
    # Non-blocking async client with no timeout limit
    http_client = httpx.AsyncClient(timeout=None)

    # Attach safe logging handler to uvicorn loggers
    safe_handler = SafeDualLogHandler(dual_logger)
    safe_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    safe_handler.setFormatter(safe_formatter)

    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        l = logging.getLogger(logger_name)
        l.addHandler(safe_handler)

    # Start background workers
    asyncio.create_task(manager.auto_unload_task())
    asyncio.create_task(heartbeat_status_task())
    asyncio.create_task(sync_llama_log_task())

    ts_print(f"📡 Shared Drive Log: {LOG_FILE}")
    ts_print(f"📁 Local Fallback Log: {LOCAL_LOG_FILE}")
    ts_print(f"📊 Status JSON: {STATUS_FILE}")
    update_status("idle")


@app.on_event("shutdown")
async def shutdown_event():
    update_status("offline")
    if http_client is not None:
        await http_client.aclose()


def _model_meta(name: str):
    if "27B" in name:
        param = "27B"
        size = 15309040064
    elif "4B" in name:
        param = "4B"
        size = 2707513696
    else:
        param = "9B"
        size = 5629109408

    if "IQ4_XS" in name or "cyber" in name.lower():
        quant = "IQ4_XS"
    elif "i1" in name.lower() or "Q5_K_M" in name:
        quant = "Q5_K_M"
    else:
        quant = "Q4_K_M"

    family = "qwen2" if "Qwen" in name or "Ornith" in name else "llama"
    return param, size, quant, family


# ── Instant Ollama & OpenAI Mock Endpoints (Served by Gateway, Non-blocking) ──────────────────
@app.get("/api/tags")
async def ollama_tags():
    models = []
    for name in manager.configs:
        param, size, quant, family = _model_meta(name)
        models.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "size": size,
            "digest": "sha256:" + "0" * 64,
            "details": {"format": "gguf", "family": family,
                        "parameter_size": param,
                        "quantization_level": quant},
        })
    return {"models": models}


def _ctx_of(cmd: str, default: int = 8192) -> int:
    if "-c " in cmd:
        try:
            return int(cmd.split("-c ")[1].split(" ")[0])
        except Exception:
            pass
    return default


@app.get("/v1/models")
@app.get("/api/v1/models")
async def openai_models():
    data = []
    for name, cmd in manager.configs.items():
        ctx = _ctx_of(cmd)
        data.append({"id": name, "object": "model", "created": int(time.time()),
                     "owned_by": "library", "max_model_len": ctx, "context_window": ctx})
    return {"object": "list", "data": data}


@app.get("/api/version")
@app.get("/version")
async def ollama_version():
    return {"version": "0.1.48"}


@app.get("/v1/props")
async def v1_props():
    return {"properties": {}}


@app.get("/health")
async def gateway_health():
    return {
        "status": "ok",
        "model": manager.current_model,
        "active_requests": manager.active_requests,
        "shared_drive_connected": dual_logger.is_primary_online,
    }


@app.post("/api/show")
async def ollama_show(request: Request):
    try:
        body = await request.json()
        model_name = body.get("name", "Ornith-1.5-9B-65k").replace(":latest", "")
    except Exception:
        model_name = "Ornith-1.5-9B-65k"
    cmd = manager.configs.get(model_name, "")
    ctx = _ctx_of(cmd)
    param, _, quant, family = _model_meta(model_name)
    return {
        "modelfile": f"FROM {model_name}\nPARAMETER num_ctx {ctx}\n",
        "parameters": f"num_ctx                        {ctx}\n",
        "template": "{{ .Prompt }}",
        "details": {"format": "gguf", "family": family,
                    "parameter_size": param,
                    "quantization_level": quant},
        "model_info": {"llama.context_length": ctx, "qwen2.context_length": ctx,
                       "general.context_length": ctx},
    }


def _recover_content_from_reasoning(resp_json: dict) -> dict:
    choices = resp_json.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        if not content and reasoning:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", reasoning, re.DOTALL)
            if not m:
                m = re.search(r"(\{[\s\S]*\})", reasoning)
            if m:
                try:
                    json.loads(m.group(1))
                    msg["content"] = m.group(1)
                except Exception:
                    msg["content"] = reasoning
            else:
                msg["content"] = reasoning
    return resp_json


# ── API Intercept & Async Non-blocking Forwarding ──────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_llm(path: str, request: Request):
    body = await request.body()
    client_ip = request.client.host if request.client else "unknown"
    start_time = time.time()

    is_chat = any(p in path for p in ("chat/completions", "api/chat", "api/generate", "v1/completions"))
    is_ollama = path in ("api/chat", "api/generate")
    is_stream = False
    req_data = {}
    if request.method in ("POST", "PUT"):
        try:
            req_data = await request.json()
            is_stream = bool(req_data.get("stream", False))
        except Exception:
            pass

    # 1. Ensure model is loaded on-demand
    counted = False
    requested_model = req_data.get("model", "Ornith-1.5-9B-65k") if isinstance(req_data, dict) else "Ornith-1.5-9B-65k"
    if is_chat and request.method == "POST":
        manager.active_requests += 1
        counted = True
        update_status("busy", {"active_requests": manager.active_requests})
        ts_print(f"📥 [{client_ip}] Incoming {path} for [{requested_model}] (stream={is_stream})")
        try:
            await manager.ensure_model(requested_model)
        except Exception as e:
            ts_print(f"Error ensuring model: {e}")
            await manager.ensure_model("Ornith-1.5-9B-65k")

    def _release():
        nonlocal counted
        if counted:
            manager.active_requests -= 1
            manager.last_active_time = time.time()
            counted = False
            status_str = "busy" if manager.active_requests > 0 else "ready"
            update_status(status_str, {"active_requests": manager.active_requests})

    # 2. Rewrite body for llama-server defaults
    target_path = path
    if request.method in ("POST", "PUT") and isinstance(req_data, dict) and req_data:
        try:
            if "model" in req_data:
                req_data["model"] = "default"
            req_data.setdefault("top_p", 0.8)
            req_data.setdefault("top_k", 20)
            req_data.setdefault("min_p", 0.0)
            req_data.setdefault("presence_penalty", 0.0)
            req_data.setdefault("repetition_penalty", 1.0)
            body = json.dumps(req_data).encode("utf-8")
        except Exception as e:
            ts_print("Error modifying request body:", e)

    if path == "api/chat":
        target_path = "v1/chat/completions"
    elif path == "api/generate":
        target_path = "v1/completions"

    url = f"{UPSTREAM}/{target_path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    # 3a. NON-STREAMING: Async forward without timeout
    if not is_stream:
        try:
            r = await http_client.request(
                request.method, url, headers=headers, content=body, params=request.query_params,
            )
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            _release()
            if isinstance(e, asyncio.CancelledError):
                ts_print(f"⚠️ [{client_ip}] Request cancelled by client")
                raise
            ts_print(f"❌ Connection to llama-server failed: {e}")
            return Response(content=json.dumps({"error": f"llama-server unreachable: {e}"}),
                            status_code=502, media_type="application/json")
        try:
            elapsed = time.time() - start_time
            if r.status_code != 200:
                ts_print(f"⚠️ [{client_ip}] Upstream returned {r.status_code} in {elapsed:.2f}s")
                return Response(content=r.content, status_code=r.status_code,
                                media_type=r.headers.get("content-type", "application/json"))
            if is_ollama:
                resp_json = r.json()
                choices = resp_json.get("choices") or [{}]
                msg = (choices[0] if choices else {}).get("message") or {}
                content = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
                ts_print(f"📤 [{client_ip}] Completed {path} in {elapsed:.2f}s")
                return JSONResponse(content={
                    "model": manager.current_model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                })
            resp_json = _recover_content_from_reasoning(r.json())
            ts_print(f"📤 [{client_ip}] Completed {path} in {elapsed:.2f}s")
            return JSONResponse(content=resp_json, status_code=r.status_code)
        except Exception:
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
        finally:
            _release()

    # 3b. STREAMING: Stream chunks asynchronously as they arrive
    async def generate():
        try:
            async with http_client.stream(
                request.method, url, headers=headers, content=body, params=request.query_params,
            ) as r:
                if r.status_code != 200:
                    yield await r.aread()
                    return
                if not is_ollama:
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            yield chunk
                else:
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            yield (json.dumps({"model": manager.current_model, "done": True}) + "\n").encode()
                            continue
                        try:
                            data = json.loads(payload)
                            if "api/chat" in path:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "") or delta.get("reasoning_content", "")
                                out = {"model": data.get("model", manager.current_model),
                                       "message": {"role": "assistant", "content": content}, "done": False}
                            else:
                                content = data["choices"][0].get("text", "")
                                out = {"model": data.get("model", manager.current_model),
                                       "response": content, "done": False}
                            yield (json.dumps(out) + "\n").encode()
                        except Exception:
                            pass
            elapsed = time.time() - start_time
            ts_print(f"📤 [{client_ip}] Stream completed {path} in {elapsed:.2f}s")
        except asyncio.CancelledError:
            ts_print(f"⚠️ [{client_ip}] Client disconnected → cancelled upstream generation")
            raise
        except httpx.HTTPError as e:
            ts_print(f"❌ Proxy streaming error: {e}")
            yield json.dumps({"error": "proxy failed"}).encode() + b"\n"
        finally:
            _release()

    return StreamingResponse(
        generate(), status_code=200,
        media_type="text/event-stream" if "v1/" in path else "application/x-ndjson",
    )


# ── Server Entrypoint ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ts_print("🔥 Starting Pentest Smart Gateway (async, non-blocking) on port 11434 (mocking Ollama)...")
    config = Config(app=app, host="0.0.0.0", port=11434, loop="asyncio",
                    lifespan="on", log_config=log_config)
    server = Server(config)
    try:
        asyncio.run(server.serve())
    except OSError as e:
        if "insufficient buffer space" in str(e) or "queue was full" in str(e):
            ts_print("⚠️ Socket buffer issue. Waiting 30s and retrying...")
            time.sleep(30)
            asyncio.run(server.serve())
        else:
            raise
