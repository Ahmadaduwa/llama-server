# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "httpx",
#     "uvicorn",
#     "python-dotenv",
# ]
# ///

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
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from uvicorn.config import Config, LOGGING_CONFIG
from uvicorn.server import Server


# ============================================================
# UTF-8
# ============================================================

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# BASE / ENV
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("API_KEY", "")

# ============================================================
# TWO LLAMA SERVERS
# ============================================================

LLM_HOST = os.getenv("LLM_HOST", "127.0.0.1")
LLM_PORT = int(os.getenv("LLM_PORT", "8080"))

EMBED_HOST = os.getenv("EMBED_HOST", "127.0.0.1")
EMBED_PORT = int(os.getenv("EMBED_PORT", "8081"))

UPSTREAM = f"http://{LLM_HOST}:{LLM_PORT}"
EMBEDDING_UPSTREAM = f"http://{EMBED_HOST}:{EMBED_PORT}"


# ============================================================
# SHARED DRIVE
# ============================================================

def _normalize_dir(d: str) -> str:
    if not d:
        return ""

    d = d.strip().replace("/", "\\")

    if d.endswith(":") and len(d) == 2:
        d += "\\"

    return d


LOG_DIR = _normalize_dir(
    os.getenv("LOG_DIR", "Z:\\")
)


def _resolve_path(
    env_var: str,
    default_filename: str,
    fallback_dir: str = ""
) -> str:

    val = os.getenv(env_var)

    if val:
        return val.strip()

    if fallback_dir:
        return os.path.join(
            fallback_dir,
            default_filename
        )

    return os.path.join(
        BASE_DIR,
        default_filename
    )


LOG_FILE = _resolve_path(
    "LOG_FILE",
    "server.log",
    LOG_DIR
)

STATUS_FILE = _resolve_path(
    "STATUS_FILE",
    "gateway_status.json",
    LOG_DIR
)

LLAMA_LOG_FILE = _resolve_path(
    "LLAMA_LOG_FILE",
    "llama-server.log",
    LOG_DIR
)

EMBED_LOG_FILE = _resolve_path(
    "EMBED_LOG_FILE",
    "embedding-server.log",
    LOG_DIR
)

LOCAL_LOG_FILE = os.getenv(
    "LOCAL_LOG_FILE",
    os.path.join(BASE_DIR, "server.log")
)

LOCAL_STATUS_FILE = os.getenv(
    "LOCAL_STATUS_FILE",
    os.path.join(BASE_DIR, "gateway_status.json")
)

LOCAL_LLAMA_LOG_FILE = os.getenv(
    "LOCAL_LLAMA_LOG_FILE",
    os.path.join(BASE_DIR, "llama-server.log")
)

LOCAL_EMBED_LOG_FILE = os.getenv(
    "LOCAL_EMBED_LOG_FILE",
    os.path.join(BASE_DIR, "embedding-server.log")
)


# ============================================================
# SAFE DUAL LOGGER
# ============================================================

class SafeDualLogger:

    def __init__(
        self,
        primary_path: str | None,
        fallback_path: str
    ):

        self.primary_path = primary_path
        self.fallback_path = fallback_path

        self._lock = threading.Lock()
        self._primary_online = True
        self._last_warn_time = 0.0

    @property
    def is_primary_online(self):
        return self._primary_online

    def write(self, message: str):

        line = (
            message
            if message.endswith("\n")
            else f"{message}\n"
        )

        with self._lock:

            # -------------------------
            # LOCAL
            # -------------------------

            try:

                if self.fallback_path:

                    f_dir = os.path.dirname(
                        self.fallback_path
                    )

                    if (
                        f_dir
                        and not os.path.exists(f_dir)
                    ):
                        os.makedirs(
                            f_dir,
                            exist_ok=True
                        )

                    with open(
                        self.fallback_path,
                        "a",
                        encoding="utf-8",
                        errors="replace"
                    ) as f:

                        f.write(line)

            except Exception:
                pass


            # -------------------------
            # SHARED DRIVE
            # -------------------------

            if self.primary_path:

                try:

                    p_dir = os.path.dirname(
                        self.primary_path
                    )

                    if (
                        p_dir
                        and not os.path.exists(p_dir)
                    ):
                        os.makedirs(
                            p_dir,
                            exist_ok=True
                        )

                    with open(
                        self.primary_path,
                        "a",
                        encoding="utf-8",
                        errors="replace"
                    ) as f:

                        f.write(line)


                    if not self._primary_online:

                        self._primary_online = True

                        now_str = datetime.now().strftime(
                            "[%Y-%m-%d %H:%M:%S]"
                        )

                        recon_msg = (
                            f"{now_str} "
                            f"🟢 [Shared Drive] "
                            f"Reconnected: "
                            f"{self.primary_path}\n"
                        )

                        try:
                            print(recon_msg.strip())
                        except Exception:
                            pass


                except Exception as e:

                    now = time.time()

                    if (
                        self._primary_online
                        or now - self._last_warn_time > 60
                    ):

                        self._primary_online = False
                        self._last_warn_time = now

                        now_str = datetime.now().strftime(
                            "[%Y-%m-%d %H:%M:%S]"
                        )

                        warn_msg = (
                            f"{now_str} "
                            f"⚠️ [Shared Drive] "
                            f"Cannot write to "
                            f"'{self.primary_path}': "
                            f"{e}. "
                            f"Using local log."
                        )

                        try:
                            print(warn_msg)
                        except Exception:
                            pass


dual_logger = SafeDualLogger(
    primary_path=LOG_FILE,
    fallback_path=LOCAL_LOG_FILE
)


def ts_print(*args, **kwargs):

    now = datetime.now().strftime(
        "[%Y-%m-%d %H:%M:%S]"
    )

    msg = " ".join(
        str(a)
        for a in args
    )

    print(
        f"{now} {msg}",
        **kwargs
    )

    dual_logger.write(
        f"{now} {msg}"
    )


# ============================================================
# UVICORN LOGGING
# ============================================================

class SafeDualLogHandler(logging.Handler):

    def __init__(self, target_logger):

        super().__init__()

        self.target_logger = target_logger

    def emit(self, record):

        try:

            msg = self.format(record)

            self.target_logger.write(msg)

        except Exception:

            self.handleError(record)


log_config = LOGGING_CONFIG.copy()

log_config["formatters"]["access"]["fmt"] = (
    '[%(asctime)s] %(levelprefix)s '
    '%(client_addr)s - "%(request_line)s" '
    '%(status_code)s'
)

log_config["formatters"]["default"]["fmt"] = (
    '[%(asctime)s] %(levelprefix)s %(message)s'
)

log_config["formatters"]["access"]["datefmt"] = (
    "%Y-%m-%d %H:%M:%S"
)

log_config["formatters"]["default"]["datefmt"] = (
    "%Y-%m-%d %H:%M:%S"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Pentest LLM Gateway"
)


# ============================================================
# HTTP CLIENT
# ============================================================

HTTP_TIMEOUT = None

http_client: httpx.AsyncClient | None = None


# ============================================================
# API KEY
# ============================================================

@app.middleware("http")
async def verify_api_key(
    request: Request,
    call_next
):

    if request.method == "OPTIONS":
        return await call_next(request)

    safe_paths = [

        "/api/tags",
        "/v1/models",
        "/api/v1/models",
        "/api/version",
        "/version",

        "/v1/props",
        "/props",

        "/api/show",

        "/health",
        "/health/llm",
        "/health/embedding",

        "/slots",
        "/metrics",

        "/vram",
        "/v1/vram",
        "/gpu",

    ]

    if request.url.path in safe_paths:
        return await call_next(request)

    if API_KEY:

        auth_header = request.headers.get(
            "Authorization"
        )

        if auth_header != f"Bearer {API_KEY}":

            return JSONResponse(
                status_code=401,
                content={
                    "error":
                    "Unauthorized: "
                    "Invalid or missing API Key"
                }
            )

    return await call_next(request)


# ============================================================
# SMART LLM MANAGER
# ============================================================

class SmartLLMManager:

    def __init__(self):

        self.current_process = None
        self.current_model = None

        self.last_active_time = time.time()

        self.active_requests = 0

        self.lock = asyncio.Lock()

        self._llama_log_handle = None


        # ----------------------------------------------------
        # PATH
        # ----------------------------------------------------

        self.base_dir = (
            "D:\\Program\\"
            "llama-b10054-bin-win-cuda-13.3-x64"
        )

        self.exe = os.path.join(
            self.base_dir,
            "llama-server.exe"
        )


        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        model_9b = os.path.join(
            self.base_dir,
            "model\\Qwen3.5-9B-GGUF\\"
            "Qwen3.5-9B-Q4_K_M.gguf"
        )

        model_9b_uncen = os.path.join(
            self.base_dir,
            "model\\"
            "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\\"
            "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
        )

        model_ornith = os.path.join(
            self.base_dir,
            "model\\"
            "Ornith-1.5-9B-uncensored\\"
            "Ornith-1.5-9B-uncensored.Q4_K_M.gguf"
        )

        model_ornith_i1 = os.path.join(
            self.base_dir,
            "model\\"
            "Ornith-1.5-9B-uncensored-i1-GGUF\\"
            "Ornith-1.5-9B-uncensored.i1-Q5_K_M.gguf"
        )

        model_ornith_35b = os.path.join(
            self.base_dir,
            "model\\"
            "Ornith-1.5-35B-A3B-abliterated-i1-GGUF\\"
            "Ornith-1.5-35B-A3B-abliterated.i1-IQ3_M.gguf"
        )

        model_27b_cyber = os.path.join(
            self.base_dir,
            "model\\"
            "Qwen3.8-27B-Uncensored-Cyber-i1-GGUF\\"
            "Qwen3.8-27B-Uncensored-Cyber.i1-IQ4_XS.gguf"
        )

        model_4b = os.path.join(
            self.base_dir,
            "model\\Qwen3.5-4B-GGUF\\"
            "Qwen3.5-4B-Q4_K_M.gguf"
        )


        # ----------------------------------------------------
        # LLM CONFIGS
        # ----------------------------------------------------

        self.configs = {

            "Ornith-1.5-35B-65k":
                f'& "{self.exe}" '
                f'-m "{model_ornith_35b}" '
                f'--port {LLM_PORT} '
                '-ngl 28 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--n-cpu-moe 27 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Ornith-1.5-35B-32k":
                f'& "{self.exe}" '
                f'-m "{model_ornith_35b}" '
                f'--port {LLM_PORT} '
                '-ngl 28 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--n-cpu-moe 27 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Ornith-1.5-35B-16k":
                f'& "{self.exe}" '
                f'-m "{model_ornith_35b}" '
                f'--port {LLM_PORT} '
                '-ngl 28 '
                '-c 16384 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--n-cpu-moe 27 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.8-27B-65k-cyber":
                f'& "{self.exe}" '
                f'-m "{model_27b_cyber}" '
                f'--port {LLM_PORT} '
                '-ngl 26 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256 '
                '--spec-type draft-mtp '
                '--spec-draft-n-max 2',


            "Qwen3.8-27B-32k-cyber":
                f'& "{self.exe}" '
                f'-m "{model_27b_cyber}" '
                f'--port {LLM_PORT} '
                '-ngl 26 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256 '
                '--spec-type draft-mtp '
                '--spec-draft-n-max 2',


            "Qwen3.8-27B-16k-cyber":
                f'& "{self.exe}" '
                f'-m "{model_27b_cyber}" '
                f'--port {LLM_PORT} '
                '-ngl 26 '
                '-c 16384 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256 '
                '--spec-type draft-mtp '
                '--spec-draft-n-max 2',


            "Ornith-1.5-9B-32k":
                f'& "{self.exe}" '
                f'-m "{model_ornith}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Ornith-1.5-9B-65k":
                f'& "{self.exe}" '
                f'-m "{model_ornith}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Ornith-1.5-9B-i1-32k":
                f'& "{self.exe}" '
                f'-m "{model_ornith_i1}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Ornith-1.5-9B-i1-65k":
                f'& "{self.exe}" '
                f'-m "{model_ornith_i1}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-32k-uncen":
                f'& "{self.exe}" '
                f'-m "{model_9b_uncen}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-65k-uncen":
                f'& "{self.exe}" '
                f'-m "{model_9b_uncen}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-132k-uncen":
                f'& "{self.exe}" '
                f'-m "{model_9b_uncen}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 131072 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-192k-uncen":
                f'& "{self.exe}" '
                f'-m "{model_9b_uncen}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 196608 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-32k":
                f'& "{self.exe}" '
                f'-m "{model_9b}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 32768 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-65k":
                f'& "{self.exe}" '
                f'-m "{model_9b}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-132k":
                f'& "{self.exe}" '
                f'-m "{model_9b}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 131072 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-9B-192k":
                f'& "{self.exe}" '
                f'-m "{model_9b}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 196608 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q4_0 '
                '-ctv q4_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',


            "Qwen3.5-4B-64k":
                f'& "{self.exe}" '
                f'-m "{model_4b}" '
                f'--port {LLM_PORT} '
                '-ngl 99 '
                '-c 65536 '
                '-b 2048 '
                '-ub 1024 '
                '--no-mmap '
                '-fa on '
                '-ctk q8_0 '
                '-ctv q8_0 '
                '-t 8 '
                '-tb 8 '
                '--parallel 1 '
                '--cont-batching '
                '--jinja '
                '--cache-reuse 256',
        }


        # ----------------------------------------------------
        # ALIASES
        # ----------------------------------------------------

        self.aliases = {

            "ornith-1.5-35b":
                "Ornith-1.5-35B-65k",

            "ornith-35b":
                "Ornith-1.5-35B-65k",

            "35b":
                "Ornith-1.5-35B-65k",

            "qwen3.8-27b":
                "Qwen3.8-27B-65k-cyber",

            "qwen-27b":
                "Qwen3.8-27B-65k-cyber",

            "27b":
                "Qwen3.8-27B-65k-cyber",

            "qwen3.5-9b":
                "Qwen3.5-9B-65k",

            "qwen-9b":
                "Qwen3.5-9B-65k",

            "9b":
                "Qwen3.5-9B-65k",
        }


        self.LOAD_TIMEOUT_SECONDS = int(
            os.getenv(
                "GATEWAY_LOAD_TIMEOUT",
                "180"
            )
        )

        self.IDLE_UNLOAD_SECONDS = int(
            os.getenv(
                "GATEWAY_IDLE_UNLOAD_SECONDS",
                "600"
            )
        )


    # ========================================================
    # HEALTH
    # ========================================================

    async def healthy(self):

        try:

            r = await http_client.get(
                f"{UPSTREAM}/health",
                timeout=2.0
            )

            return r.status_code == 200

        except Exception:
            return False


    # ========================================================
    # KILL LLM SERVER ONLY
    # ========================================================

    @staticmethod
    def kill_llm_servers():

        try:

            subprocess.run(
                [
                    "powershell",
                    "-Command",

                    f"""
                    Get-NetTCPConnection `
                        -LocalPort {LLM_PORT} `
                        -ErrorAction SilentlyContinue |
                    ForEach-Object {{
                        Stop-Process `
                            -Id $_.OwningProcess `
                            -Force `
                            -ErrorAction SilentlyContinue
                    }}
                    """
                ],
                check=False
            )

        except Exception as e:

            ts_print(
                f"⚠️ LLM kill failed: {e}"
            )


    # ========================================================
    # LOG TAIL
    # ========================================================

    @staticmethod
    def log_tail(
        n=30
    ):

        try:

            if not os.path.exists(
                LOCAL_LLAMA_LOG_FILE
            ):
                return "(no llama log)"

            with open(
                LOCAL_LLAMA_LOG_FILE,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                lines = f.readlines()

            return "".join(
                lines[-n:]
            ).strip()

        except Exception as e:

            return str(e)


    # ========================================================
    # DOWNGRADE
    # ========================================================

    def downgrade(
        self,
        model_name
    ):

        m = re.match(
            r"^(.*?)-(\d+)k(.*)$",
            model_name or ""
        )

        if not m:
            return None

        prefix = m.group(1)
        ctx = int(m.group(2))
        suffix = m.group(3) or ""

        candidates = []

        for key in self.configs:

            mm = re.match(
                rf"^{re.escape(prefix)}-(\d+)k{re.escape(suffix)}$",
                key
            )

            if mm:

                candidates.append(
                    int(mm.group(1))
                )

        candidates.sort(
            reverse=True
        )

        smaller = next(
            (
                c
                for c in candidates
                if c < ctx
            ),
            None
        )

        if smaller is None:
            return None

        return (
            f"{prefix}-"
            f"{smaller}k"
            f"{suffix}"
        )


    # ========================================================
    # ENSURE MODEL
    # ========================================================

    async def ensure_model(
        self,
        model_name
    ):

        async with self.lock:

            clean_name = (
                model_name or ""
            )

            clean_name = (
                clean_name
                .replace("openai/", "")
                .replace(":latest", "")
                .strip()
            )

            low = clean_name.lower()

            if low in self.aliases:

                clean_name = (
                    self.aliases[low]
                )


            if clean_name not in self.configs:

                case_map = {
                    k.lower(): k
                    for k in self.configs
                }

                if (
                    clean_name.lower()
                    in case_map
                ):

                    clean_name = case_map[
                        clean_name.lower()
                    ]

                else:

                    ts_print(
                        f"⚠️ Unknown model "
                        f"'{model_name}' "
                        f"→ defaulting to "
                        f"Qwen3.5-9B-65k"
                    )

                    clean_name = (
                        "Qwen3.5-9B-65k"
                    )


            model_name = clean_name


            # ------------------------------------------------
            # Already loaded
            # ------------------------------------------------

            if (
                self.current_model
                == model_name
                and self.current_process
            ):

                proc_dead = (
                    self.current_process.poll()
                    is not None
                )

                if (
                    not proc_dead
                    and await self.healthy()
                ):

                    self.last_active_time = (
                        time.time()
                    )

                    return


                self.unload()


            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            self.unload()

            while True:

                ts_print(
                    f"🚀 Loading LLM "
                    f"[{model_name}] "
                    f"on port {LLM_PORT}"
                )

                update_status(
                    "loading",
                    {
                        "loading_model":
                        model_name
                    }
                )


                self.kill_llm_servers()

                await asyncio.sleep(1)


                cmd = self.configs[
                    model_name
                ]


                # Log

                try:

                    if (
                        self._llama_log_handle
                        and not self._llama_log_handle.closed
                    ):

                        self._llama_log_handle.close()

                except Exception:
                    pass


                try:

                    self._llama_log_handle = open(
                        LOCAL_LLAMA_LOG_FILE,
                        "a",
                        encoding="utf-8",
                        errors="replace",
                        buffering=1
                    )

                    stdout_target = (
                        self._llama_log_handle
                    )

                except Exception:

                    stdout_target = (
                        subprocess.DEVNULL
                    )


                # Launch

                self.current_process = (
                    subprocess.Popen(
                        [
                            "powershell",
                            "-Command",
                            cmd
                        ],
                        stdout=stdout_target,
                        stderr=subprocess.STDOUT
                    )
                )


                self.current_model = (
                    model_name
                )

                self.last_active_time = (
                    time.time()
                )


                start = time.time()

                deadline = (
                    start
                    + self.LOAD_TIMEOUT_SECONDS
                )

                ready = False
                crashed = False
                exit_code = None


                while time.time() < deadline:

                    exit_code = (
                        self.current_process.poll()
                    )

                    if exit_code is not None:

                        crashed = True
                        break


                    if await self.healthy():

                        ready = True
                        break


                    await asyncio.sleep(1)


                # ------------------------------------------------
                # READY
                # ------------------------------------------------

                if ready:

                    ts_print(
                        f"✅ LLM "
                        f"[{model_name}] ready "
                        f"in "
                        f"{int(time.time() - start)}s"
                    )

                    update_status(
                        "ready",
                        {
                            "current_model":
                            model_name
                        }
                    )

                    break


                elapsed = int(
                    time.time() - start
                )


                # ------------------------------------------------
                # CRASH
                # ------------------------------------------------

                if crashed:

                    ts_print(
                        f"❌ LLM "
                        f"[{model_name}] "
                        f"EXITED after "
                        f"{elapsed}s "
                        f"(exit={exit_code})\n"
                        f"──── llama log ────\n"
                        f"{self.log_tail()}\n"
                        f"──────────────────"
                    )

                else:

                    ts_print(
                        f"⚠️ LLM "
                        f"[{model_name}] "
                        f"not ready after "
                        f"{self.LOAD_TIMEOUT_SECONDS}s"
                    )


                self.unload()


                fast_crash = (
                    crashed
                    and elapsed < 20
                )


                smaller = (
                    None
                    if fast_crash
                    else self.downgrade(
                        model_name
                    )
                )


                if (
                    smaller
                    and smaller in self.configs
                ):

                    ts_print(
                        f"⚠️ Downgrade "
                        f"{model_name} "
                        f"→ "
                        f"{smaller}"
                    )

                    model_name = smaller

                    continue


                self.current_model = None

                update_status(
                    "error",
                    {
                        "error":
                        f"Failed to load "
                        f"{model_name}"
                    }
                )

                break


    # ========================================================
    # UNLOAD LLM
    # ========================================================

    def unload(self):

        if (
            self.current_process
            or self._llama_log_handle
        ):

            ts_print(
                "💤 Unloading LLM "
                f"from port {LLM_PORT}"
            )

            self.kill_llm_servers()


            if self.current_process:

                try:

                    self.current_process.terminate()

                    self.current_process.wait(
                        timeout=1
                    )

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

            update_status(
                "idle",
                {
                    "current_model":
                    None
                }
            )


    # ========================================================
    # AUTO UNLOAD
    # ========================================================

    async def auto_unload_task(self):

        idle = (
            self.IDLE_UNLOAD_SECONDS
        )

        mins = idle // 60

        while True:

            await asyncio.sleep(30)

            if (
                self.current_process
                and self.active_requests == 0
                and (
                    time.time()
                    - self.last_active_time
                    > idle
                )
            ):

                async with self.lock:

                    if (
                        self.current_process
                        and self.active_requests == 0
                        and (
                            time.time()
                            - self.last_active_time
                            > idle
                        )
                    ):

                        ts_print(
                            f"⏳ LLM idle "
                            f"> {mins} min "
                            f"→ unloading"
                        )

                        self.unload()


manager = SmartLLMManager()


# ============================================================
# EMBEDDING SERVER MANAGER
# ============================================================

class EmbeddingServerManager:

    def __init__(self):

        self.process = None
        self.lock = asyncio.Lock()

        self.base_dir = (
            "D:\\Program\\"
            "llama-b10054-bin-win-cuda-13.3-x64"
        )

        self.exe = os.path.join(
            self.base_dir,
            "llama-server.exe"
        )


        # ----------------------------------------------------
        # NOMIC V2
        # ----------------------------------------------------

        self.model = os.path.join(
            self.base_dir,
            "model\\"
            "nomic-embed-text-v1.5-GGUF\\"
            "nomic-embed-text-v1.5.Q8_0.gguf"
        )


    # ========================================================
    # HEALTH
    # ========================================================

    async def healthy(self):

        try:

            r = await http_client.get(
                f"{EMBEDDING_UPSTREAM}/health",
                timeout=2.0
            )

            return r.status_code == 200

        except Exception:

            return False


    # ========================================================
    # KILL EMBEDDING SERVER ONLY
    # ========================================================

    @staticmethod
    def kill_embedding_server():

        try:

            subprocess.run(
                [
                    "powershell",
                    "-Command",

                    f"""
                    Get-NetTCPConnection `
                        -LocalPort {EMBED_PORT} `
                        -ErrorAction SilentlyContinue |
                    ForEach-Object {{
                        Stop-Process `
                            -Id $_.OwningProcess `
                            -Force `
                            -ErrorAction SilentlyContinue
                    }}
                    """
                ],
                check=False
            )

        except Exception as e:

            ts_print(
                f"⚠️ Embedding kill failed: {e}"
            )


    # ========================================================
    # START
    # ========================================================

    async def ensure_started(self):

        async with self.lock:

            if await self.healthy():

                return True


            self.kill_embedding_server()

            await asyncio.sleep(1)


            if not os.path.exists(
                self.model
            ):

                ts_print(
                    "❌ Embedding model "
                    "not found:"
                )

                ts_print(
                    self.model
                )

                return False


            try:

                log_handle = open(
                    LOCAL_EMBED_LOG_FILE,
                    "a",
                    encoding="utf-8",
                    errors="replace",
                    buffering=1
                )

            except Exception:

                log_handle = (
                    subprocess.DEVNULL
                )


            # ------------------------------------------------
            # IMPORTANT
            #
            # --embedding
            # --pooling cls
            #
            # Nomic model is an embedding model.
            # ------------------------------------------------

            cmd = (

                f'& "{self.exe}" '

                f'-m "{self.model}" '

                f'--port {EMBED_PORT} '

                '-ngl 99 '

                '-c 512 '

                '-b 1024 '

                '-ub 1024 '

                '--embedding '

                '--pooling cls '

                '--parallel 4 '

                '--cont-batching'
            )


            ts_print(
                "🧠 Starting embedding "
                "llama-server"
            )

            ts_print(
                f"   Model: {self.model}"
            )

            ts_print(
                f"   Port: {EMBED_PORT}"
            )


            self.process = subprocess.Popen(
                [
                    "powershell",
                    "-Command",
                    cmd
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT
            )


            deadline = (
                time.time()
                + 120
            )


            while time.time() < deadline:

                if await self.healthy():

                    ts_print(
                        "✅ Embedding server "
                        f"ready on :{EMBED_PORT}"
                    )

                    return True


                if (
                    self.process.poll()
                    is not None
                ):

                    ts_print(
                        "❌ Embedding server "
                        "crashed"
                    )

                    return False


                await asyncio.sleep(1)


            ts_print(
                "❌ Embedding server "
                "startup timeout"
            )

            return False


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        ts_print(
            "🛑 Stopping embedding "
            "llama-server"
        )

        self.kill_embedding_server()

        self.process = None


embedding_manager = (
    EmbeddingServerManager()
)


# ============================================================
# STATUS
# ============================================================

def safe_write_json(
    file_path,
    data
):

    if not file_path:
        return

    try:

        p_dir = os.path.dirname(
            file_path
        )

        if (
            p_dir
            and not os.path.exists(p_dir)
        ):

            os.makedirs(
                p_dir,
                exist_ok=True
            )

        content = json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )

        with open(
            file_path,
            "w",
            encoding="utf-8",
            errors="replace"
        ) as f:

            f.write(content)

    except Exception:
        pass


def update_status(
    status,
    extra=None
):

    data = {

        "status": status,

        "current_model":
            manager.current_model,

        "active_requests":
            manager.active_requests,

        "embedding_server":
            embedding_manager.process
            is not None,

        "llm_upstream":
            UPSTREAM,

        "embedding_upstream":
            EMBEDDING_UPSTREAM,

        "last_heartbeat":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "timestamp_epoch":
            time.time(),

        "gateway_url":
            "http://100.119.233.96:11434",

        "shared_drive_connected":
            dual_logger.is_primary_online,

    }


    if extra:
        data.update(extra)


    safe_write_json(
        LOCAL_STATUS_FILE,
        data
    )

    if STATUS_FILE:

        safe_write_json(
            STATUS_FILE,
            data
        )


# ============================================================
# MODEL METADATA
# ============================================================

def _ctx_of(
    cmd,
    default=8192
):

    if "-c " in cmd:

        try:

            return int(
                cmd
                .split("-c ")[1]
                .split(" ")[0]
            )

        except Exception:
            pass

    return default


def _model_meta(name):

    if "35B" in name:

        param = "35B"
        size = 17500000000

    elif "27B" in name:

        param = "27B"
        size = 15309040064

    elif "4B" in name:

        param = "4B"
        size = 2707513696

    else:

        param = "9B"
        size = 5629109408


    if (
        "IQ3_M" in name
        or "iq3" in name.lower()
        or "35B" in name
    ):

        quant = "IQ3_M"

    elif (
        "IQ4_XS" in name
        or "cyber" in name.lower()
    ):

        quant = "IQ4_XS"

    elif (
        "i1" in name.lower()
        or "Q5_K_M" in name
    ):

        quant = "Q5_K_M"

    else:

        quant = "Q4_K_M"


    family = (
        "qwen2"
        if (
            "Qwen" in name
            or "Ornith" in name
        )
        else "llama"
    )


    return (
        param,
        size,
        quant,
        family
    )


# ============================================================
# OLLAMA TAGS
# ============================================================

@app.get("/api/tags")
async def ollama_tags():

    models = []

    for name in manager.configs:

        param, size, quant, family = (
            _model_meta(name)
        )

        models.append({

            "name":
                f"{name}:latest",

            "model":
                f"{name}:latest",

            "modified_at":
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z",
                    time.gmtime()
                ),

            "size":
                size,

            "digest":
                "sha256:" + "0" * 64,

            "details": {

                "format":
                    "gguf",

                "family":
                    family,

                "parameter_size":
                    param,

                "quantization_level":
                    quant,

            }

        })


    # Add embedding model

    models.append({

        "name":
            "nomic-embed-text-v1.5:latest",

        "model":
            "nomic-embed-text-v1.5:latest",

        "modified_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%S.000Z",
                time.gmtime()
            ),

        "size":
            390000000,

        "digest":
            "sha256:" + "1" * 64,

        "details": {

            "format":
                "gguf",

            "family":
                "nomic-bert-moe",

            "parameter_size":
                "0.5B",

            "quantization_level":
                "Q5_K_M",

        }

    })


    return {
        "models": models
    }


# ============================================================
# OPENAI MODELS
# ============================================================

@app.get("/v1/models")
@app.get("/api/v1/models")
async def openai_models():

    data = []


    for name, cmd in manager.configs.items():

        ctx = _ctx_of(cmd)

        data.append({

            "id":
                name,

            "object":
                "model",

            "created":
                int(time.time()),

            "owned_by":
                "library",

            "max_model_len":
                ctx,

            "context_window":
                ctx,

        })


    # Embedding

    data.append({

        "id":
            "nomic-embed-text-v1.5",

        "object":
            "model",

        "created":
            int(time.time()),

        "owned_by":
            "nomic-ai",

        "max_model_len":
            512,

        "context_window":
            512,

    })


    return {

        "object":
            "list",

        "data":
            data

    }


# ============================================================
# VERSION
# ============================================================

@app.get("/api/version")
@app.get("/version")
async def version():

    return {
        "version": "0.1.48"
    }


@app.get("/v1/props")
async def v1_props():

    return {
        "properties": {}
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def gateway_health():

    return {

        "status":
            "ok",

        "model":
            manager.current_model,

        "active_requests":
            manager.active_requests,

        "llm":
            await manager.healthy(),

        "embedding":
            await embedding_manager.healthy(),

        "shared_drive_connected":
            dual_logger.is_primary_online,

    }


@app.get("/health/llm")
async def llm_health():

    return {

        "status":
            "ok"
            if await manager.healthy()
            else "down",

        "model":
            manager.current_model,

        "upstream":
            UPSTREAM,

    }


@app.get("/health/embedding")
async def embedding_health():

    healthy = (
        await embedding_manager.healthy()
    )

    return {

        "status":
            "ok"
            if healthy
            else "down",

        "model":
            "nomic-embed-text-v1.5",

        "upstream":
            EMBEDDING_UPSTREAM,

    }


# ============================================================
# EMBEDDING PROXY
# ============================================================

@app.post("/v1/embeddings")
async def embeddings_proxy(
    request: Request
):

    body = await request.body()

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    start = time.time()


    try:

        req_data = await request.json()

    except Exception:

        req_data = {}


    requested_model = (
        req_data.get(
            "model",
            "nomic-embed-text-v1.5"
        )
        if isinstance(req_data, dict)
        else "nomic-embed-text-v1.5"
    )


    # --------------------------------------------------------
    # Always use Nomic embedding model
    # --------------------------------------------------------

    if isinstance(req_data, dict):

        req_data["model"] = (
            "nomic-embed-text-v1.5"
        )

        body = json.dumps(
            req_data
        ).encode("utf-8")


    ts_print(
        f"📥 [{client_ip}] "
        f"Embedding request "
        f"[{requested_model}]"
    )


    # --------------------------------------------------------
    # Ensure embedding server
    # --------------------------------------------------------

    if not await embedding_manager.ensure_started():

        return JSONResponse(

            status_code=503,

            content={

                "error": {

                    "message":
                        "Embedding server unavailable",

                    "type":
                        "server_error",

                }

            }

        )


    headers = {

        k: v

        for k, v
        in request.headers.items()

        if k.lower()
        not in (
            "host",
            "content-length"
        )

    }


    try:

        r = await http_client.post(

            f"{EMBEDDING_UPSTREAM}/v1/embeddings",

            headers=headers,

            content=body,

            timeout=None,

        )


        elapsed = (
            time.time()
            - start
        )


        ts_print(
            f"📤 [{client_ip}] "
            f"Embedding completed "
            f"in {elapsed:.2f}s "
            f"status={r.status_code}"
        )


        return Response(

            content=r.content,

            status_code=r.status_code,

            media_type=r.headers.get(
                "content-type",
                "application/json"
            )

        )


    except asyncio.CancelledError:

        raise


    except httpx.HTTPError as e:

        ts_print(
            f"❌ Embedding upstream "
            f"failed: {e}"
        )

        return JSONResponse(

            status_code=502,

            content={

                "error": {

                    "message":
                        f"Embedding server "
                        f"unreachable: {e}",

                    "type":
                        "upstream_error",

                }

            }

        )


# ============================================================
# OLLAMA EMBEDDING COMPATIBILITY
# ============================================================

@app.post("/api/embeddings")
async def ollama_embeddings(
    request: Request
):

    body = await request.body()

    try:

        req = json.loads(
            body.decode("utf-8")
        )

    except Exception:

        req = {}


    # Ollama old API:
    #
    # {
    #   "model": "nomic-embed-text",
    #   "prompt": "hello"
    # }


    prompt = req.get(
        "prompt",
        ""
    )


    # Convert to OpenAI embedding format

    openai_body = {

        "model":
            "nomic-embed-text-v1.5",

        "input":
            prompt,

    }


    if isinstance(
        prompt,
        list
    ):

        openai_body["input"] = prompt


    headers = {

        k: v

        for k, v
        in request.headers.items()

        if k.lower()
        not in (
            "host",
            "content-length"
        )

    }


    if not await embedding_manager.ensure_started():

        return JSONResponse(

            status_code=503,

            content={
                "error":
                    "Embedding server unavailable"
            }

        )


    try:

        r = await http_client.post(

            f"{EMBEDDING_UPSTREAM}/v1/embeddings",

            headers=headers,

            json=openai_body,

            timeout=None

        )


        data = r.json()


        if r.status_code != 200:

            return Response(

                content=r.content,

                status_code=r.status_code,

                media_type="application/json"

            )


        embeddings = [

            item["embedding"]

            for item
            in data.get(
                "data",
                []
            )

        ]


        # Ollama-compatible response

        if len(embeddings) == 1:

            return JSONResponse({

                "model":
                    "nomic-embed-text",

                "embedding":
                    embeddings[0]

            })


        return JSONResponse({

            "model":
                "nomic-embed-text",

            "embeddings":
                embeddings

        })


    except Exception as e:

        ts_print(
            f"❌ Ollama embedding "
            f"adapter error: {e}"
        )

        return JSONResponse(

            status_code=502,

            content={
                "error": str(e)
            }

        )


# ============================================================
# OLLAMA SHOW
# ============================================================

@app.post("/api/show")
async def ollama_show(
    request: Request
):

    try:

        body = await request.json()

        model_name = (
            body.get(
                "name",
                "Ornith-1.5-9B-65k"
            )
            .replace(
                ":latest",
                ""
            )
        )

    except Exception:

        model_name = (
            "Ornith-1.5-9B-65k"
        )


    if (
        "nomic" in
        model_name.lower()
    ):

        return {

            "modelfile":
                "FROM "
                "nomic-embed-text-v1.5\n",

            "parameters":
                "embedding\n",

            "template":
                "",

            "details": {

                "format":
                    "gguf",

                "family":
                    "nomic-bert-moe",

                "parameter_size":
                    "0.5B",

                "quantization_level":
                    "Q5_K_M",

            },

            "model_info": {

                "context_length":
                    512

            }

        }


    cmd = manager.configs.get(
        model_name,
        ""
    )

    ctx = _ctx_of(cmd)

    param, _, quant, family = (
        _model_meta(model_name)
    )


    return {

        "modelfile":
            f"FROM {model_name}\n"
            f"PARAMETER num_ctx {ctx}\n",

        "parameters":
            f"num_ctx {ctx}\n",

        "template":
            "{{ .Prompt }}",

        "details": {

            "format":
                "gguf",

            "family":
                family,

            "parameter_size":
                param,

            "quantization_level":
                quant,

        },

        "model_info": {

            "llama.context_length":
                ctx,

            "qwen2.context_length":
                ctx,

            "general.context_length":
                ctx,

        }

    }


# ============================================================
# VRAM
# ============================================================

_vram_cache = {}
_vram_cache_time = 0.0


def get_vram_info():

    global \
        _vram_cache, \
        _vram_cache_time


    now = time.time()


    if (
        _vram_cache
        and now - _vram_cache_time < 1
    ):

        return _vram_cache


    try:

        cmd = [

            "nvidia-smi",

            "--query-gpu="
            "memory.used,"
            "memory.total,"
            "utilization.gpu",

            "--format="
            "csv,nounits,noheader"

        ]


        out = subprocess.check_output(

            cmd,

            encoding="utf-8",

            timeout=1,

            stderr=subprocess.DEVNULL

        )


        line = (
            out
            .strip()
            .splitlines()[0]
        )


        parts = [

            x.strip()

            for x in
            line.split(",")

        ]


        used_mb = float(
            parts[0]
        )

        total_mb = float(
            parts[1]
        )

        util_pct = float(
            parts[2]
        )


        _vram_cache = {

            "used_mb":
                round(
                    used_mb,
                    1
                ),

            "total_mb":
                round(
                    total_mb,
                    1
                ),

            "pct":
                round(
                    used_mb
                    / total_mb
                    * 100,
                    1
                ),

            "gpu_util_pct":
                round(
                    util_pct,
                    1
                )

        }


        _vram_cache_time = now


        return _vram_cache


    except Exception:

        return {}


@app.get("/vram")
@app.get("/v1/vram")
async def gateway_vram():

    return get_vram_info()


# ============================================================
# REASONING RECOVERY
# ============================================================

def _recover_content_from_reasoning(
    resp_json
):

    choices = (
        resp_json.get("choices")
        or []
    )

    reasoning_tokens = 0


    if (
        choices
        and isinstance(
            choices[0],
            dict
        )
    ):

        msg = (
            choices[0]
            .get("message")
            or {}
        )


        content = (
            msg.get("content")
            or ""
        ).strip()


        reasoning = (
            msg.get(
                "reasoning_content"
            )
            or msg.get(
                "reasoning"
            )
            or ""
        ).strip()


        if "<think>" in content:

            m_think = re.search(
                r"<think>([\s\S]*?)"
                r"(?:</think>|$)",
                content
            )

            if m_think:

                extracted = (
                    m_think
                    .group(1)
                    .strip()
                )

                if not reasoning:
                    reasoning = extracted


                content = re.sub(
                    r"<think>[\s\S]*?</think>",
                    "",
                    content
                ).strip()


                msg["content"] = content

                msg[
                    "reasoning_content"
                ] = reasoning


        if reasoning:

            reasoning_tokens = max(
                1,
                int(
                    len(reasoning)
                    / 3.5
                )
            )


        if (
            not content
            and reasoning
        ):

            msg["content"] = (
                reasoning
            )


    usage = resp_json.setdefault(
        "usage",
        {}
    )


    if isinstance(
        usage,
        dict
    ):

        vram = get_vram_info()

        if vram:

            usage["vram"] = vram


        if reasoning_tokens > 0:

            details = (
                usage.setdefault(
                    "completion_tokens_details",
                    {}
                )
            )

            if isinstance(
                details,
                dict
            ):

                details[
                    "reasoning_tokens"
                ] = reasoning_tokens


            comp = (
                usage.get(
                    "completion_tokens"
                )
                or 0
            )


            if comp < reasoning_tokens:

                usage[
                    "completion_tokens"
                ] = (
                    comp
                    + reasoning_tokens
                )


    return resp_json


# ============================================================
# GENERATION PROXY
# ============================================================

@app.api_route(
    "/{path:path}",
    methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE"
    ]
)
async def proxy_to_llm(
    path: str,
    request: Request
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Embeddings are handled ABOVE.
    # This route is ONLY LLM generation.
    # --------------------------------------------------------

    body = await request.body()

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    start_time = time.time()


    is_chat = any(

        p in path

        for p in (

            "chat/completions",

            "api/chat",

            "api/generate",

            "v1/completions",

        )

    )


    is_ollama = path in (

        "api/chat",

        "api/generate"

    )


    is_stream = False

    req_data = {}


    if request.method in (
        "POST",
        "PUT"
    ):

        try:

            req_data = await request.json()

            is_stream = bool(
                req_data.get(
                    "stream",
                    False
                )
            )

        except Exception:

            pass


    # --------------------------------------------------------
    # Load LLM ONLY for generation
    # --------------------------------------------------------

    counted = False


    requested_model = (

        req_data.get(
            "model",
            "Ornith-1.5-9B-65k"
        )

        if isinstance(
            req_data,
            dict
        )

        else
            "Ornith-1.5-9B-65k"

    )


    if (
        is_chat
        and request.method == "POST"
    ):

        manager.active_requests += 1

        counted = True

        update_status(
            "busy",
            {
                "active_requests":
                    manager.active_requests
            }
        )


        ts_print(
            f"📥 [{client_ip}] "
            f"Incoming {path} "
            f"for [{requested_model}] "
            f"(stream={is_stream})"
        )


        try:

            await manager.ensure_model(
                requested_model
            )

        except Exception as e:

            ts_print(
                f"❌ Error ensuring model: "
                f"{e}"
            )

            await manager.ensure_model(
                "Ornith-1.5-9B-65k"
            )


    def _release():

        nonlocal counted

        if counted:

            manager.active_requests -= 1

            manager.last_active_time = (
                time.time()
            )

            counted = False


            status_str = (

                "busy"

                if manager.active_requests > 0

                else "ready"

            )


            update_status(

                status_str,

                {
                    "active_requests":
                        manager.active_requests
                }

            )


    # --------------------------------------------------------
    # Rewrite body
    # --------------------------------------------------------

    target_path = path


    if (
        request.method in (
            "POST",
            "PUT"
        )
        and isinstance(
            req_data,
            dict
        )
        and req_data
    ):

        try:

            if "model" in req_data:

                req_data["model"] = (
                    "default"
                )


            req_data.setdefault(
                "top_p",
                0.8
            )

            req_data.setdefault(
                "top_k",
                20
            )

            req_data.setdefault(
                "min_p",
                0.0
            )

            req_data.setdefault(
                "presence_penalty",
                0.0
            )

            req_data.setdefault(
                "repetition_penalty",
                1.0
            )


            body = json.dumps(
                req_data
            ).encode("utf-8")


        except Exception as e:

            ts_print(
                "⚠️ Error modifying body:",
                e
            )


    if path == "api/chat":

        target_path = (
            "v1/chat/completions"
        )


    elif path == "api/generate":

        target_path = (
            "v1/completions"
        )


    url = (
        f"{UPSTREAM}/"
        f"{target_path}"
    )


    headers = {

        k: v

        for k, v
        in request.headers.items()

        if k.lower()
        not in (
            "host",
            "content-length"
        )

    }


    # --------------------------------------------------------
    # NON STREAM
    # --------------------------------------------------------

    if not is_stream:

        try:

            r = await http_client.request(

                request.method,

                url,

                headers=headers,

                content=body,

                params=request.query_params,

            )


        except (
            httpx.HTTPError,
            asyncio.CancelledError
        ) as e:

            _release()


            if isinstance(
                e,
                asyncio.CancelledError
            ):

                raise


            ts_print(
                f"❌ LLM connection failed: "
                f"{e}"
            )


            return JSONResponse(

                status_code=502,

                content={

                    "error":
                        f"llama-server "
                        f"unreachable: {e}"

                }

            )


        try:

            elapsed = (
                time.time()
                - start_time
            )


            if r.status_code != 200:

                ts_print(

                    f"⚠️ [{client_ip}] "
                    f"LLM returned "
                    f"{r.status_code} "
                    f"in "
                    f"{elapsed:.2f}s"

                )


                return Response(

                    content=r.content,

                    status_code=
                        r.status_code,

                    media_type=
                        r.headers.get(
                            "content-type",
                            "application/json"
                        )

                )


            if is_ollama:

                resp_json = r.json()

                choices = (
                    resp_json.get(
                        "choices"
                    )
                    or [{}]
                )


                msg = (
                    choices[0]
                    .get("message")
                    or {}
                )


                content = (

                    msg.get(
                        "content"
                    )
                    or ""

                ).strip()


                if not content:

                    content = (

                        msg.get(
                            "reasoning_content"
                        )
                        or ""
                    ).strip()


                ts_print(

                    f"📤 [{client_ip}] "
                    f"Completed "
                    f"{path} "
                    f"in "
                    f"{elapsed:.2f}s"

                )


                return JSONResponse({

                    "model":
                        manager.current_model,

                    "created_at":
                        time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime()
                        ),

                    "message": {

                        "role":
                            "assistant",

                        "content":
                            content

                    },

                    "done":
                        True

                })


            resp_json = (
                _recover_content_from_reasoning(
                    r.json()
                )
            )


            ts_print(

                f"📤 [{client_ip}] "
                f"Completed "
                f"{path} "
                f"in "
                f"{elapsed:.2f}s"

            )


            return JSONResponse(

                content=resp_json,

                status_code=r.status_code

            )


        except Exception:

            return Response(

                content=r.content,

                status_code=r.status_code,

                media_type="application/json"

            )


        finally:

            _release()


    # --------------------------------------------------------
    # STREAMING
    # --------------------------------------------------------

    async def generate():

        try:

            async with http_client.stream(

                request.method,

                url,

                headers=headers,

                content=body,

                params=request.query_params,

            ) as r:

                if r.status_code != 200:

                    yield await r.aread()

                    return


                if not is_ollama:

                    async for chunk in (
                        r.aiter_bytes()
                    ):

                        if chunk:

                            yield chunk


                else:

                    async for line in (
                        r.aiter_lines()
                    ):

                        if (
                            not line
                            or not line.startswith(
                                "data: "
                            )
                        ):

                            continue


                        payload = line[6:]


                        if (
                            payload.strip()
                            == "[DONE]"
                        ):

                            yield (

                                json.dumps({
                                    "model":
                                        manager.current_model,
                                    "done":
                                        True
                                })

                                + "\n"

                            ).encode()

                            continue


                        try:

                            data = json.loads(
                                payload
                            )


                            if "api/chat" in path:

                                delta = (
                                    data[
                                        "choices"
                                    ][0]
                                    .get(
                                        "delta",
                                        {}
                                    )
                                )


                                content = (

                                    delta.get(
                                        "content",
                                        ""
                                    )

                                    or

                                    delta.get(
                                        "reasoning_content",
                                        ""
                                    )

                                )


                                out = {

                                    "model":
                                        data.get(
                                            "model",
                                            manager.current_model
                                        ),

                                    "message": {

                                        "role":
                                            "assistant",

                                        "content":
                                            content

                                    },

                                    "done":
                                        False

                                }


                            else:

                                content = (
                                    data[
                                        "choices"
                                    ][0]
                                    .get(
                                        "text",
                                        ""
                                    )
                                )


                                out = {

                                    "model":
                                        data.get(
                                            "model",
                                            manager.current_model
                                        ),

                                    "response":
                                        content,

                                    "done":
                                        False

                                }


                            yield (

                                json.dumps(
                                    out
                                )
                                + "\n"

                            ).encode()


                        except Exception:

                            pass


            elapsed = (
                time.time()
                - start_time
            )


            ts_print(

                f"📤 [{client_ip}] "
                f"Stream completed "
                f"{path} "
                f"in "
                f"{elapsed:.2f}s"

            )


        except asyncio.CancelledError:

            ts_print(

                f"⚠️ [{client_ip}] "
                f"Client disconnected "
                f"→ cancelled LLM"

            )

            raise


        except httpx.HTTPError as e:

            ts_print(
                f"❌ LLM streaming error: {e}"
            )

            yield (

                json.dumps({

                    "error":
                        "proxy failed"

                })

                .encode()

                + b"\n"

            )


        finally:

            _release()


    return StreamingResponse(

        generate(),

        status_code=200,

        media_type=(

            "text/event-stream"

            if "v1/" in path

            else
            "application/x-ndjson"

        )

    )


# ============================================================
# HEARTBEAT
# ============================================================

async def heartbeat_status_task():

    while True:

        await asyncio.sleep(10)

        try:

            status_str = (

                "busy"

                if manager.active_requests > 0

                else (

                    "ready"

                    if manager.current_model

                    else
                    "idle"
                )

            )


            update_status(
                status_str
            )

        except Exception:

            pass


# ============================================================
# SYNC LLAMA LOG
# ============================================================

async def sync_llama_log_task():

    read_pos = 0


    if os.path.exists(
        LOCAL_LLAMA_LOG_FILE
    ):

        try:

            read_pos = os.path.getsize(
                LOCAL_LLAMA_LOG_FILE
            )

        except Exception:

            read_pos = 0


    while True:

        await asyncio.sleep(2)


        if (
            not LLAMA_LOG_FILE
            or not os.path.exists(
                LOCAL_LLAMA_LOG_FILE
            )
        ):

            continue


        try:

            current_size = (
                os.path.getsize(
                    LOCAL_LLAMA_LOG_FILE
                )
            )


            if current_size > read_pos:

                with open(
                    LOCAL_LLAMA_LOG_FILE,
                    "r",
                    encoding="utf-8",
                    errors="replace"
                ) as lf:

                    lf.seek(
                        read_pos
                    )

                    new_data = lf.read()

                    read_pos = lf.tell()


                if new_data:

                    l_dir = (
                        os.path.dirname(
                            LLAMA_LOG_FILE
                        )
                    )


                    if (
                        l_dir
                        and not os.path.exists(
                            l_dir
                        )
                    ):

                        os.makedirs(
                            l_dir,
                            exist_ok=True
                        )


                    with open(
                        LLAMA_LOG_FILE,
                        "a",
                        encoding="utf-8",
                        errors="replace"
                    ) as zf:

                        zf.write(
                            new_data
                        )


            elif current_size < read_pos:

                read_pos = 0


        except Exception:

            pass


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app):

    global http_client


    http_client = httpx.AsyncClient(
        timeout=None
    )


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    safe_handler = (
        SafeDualLogHandler(
            dual_logger
        )
    )


    safe_formatter = logging.Formatter(

        "[%(asctime)s] "
        "%(levelname)s: "
        "%(message)s",

        datefmt="%Y-%m-%d %H:%M:%S"

    )


    safe_handler.setFormatter(
        safe_formatter
    )


    for logger_name in (

        "uvicorn",
        "uvicorn.access",
        "uvicorn.error"

    ):

        logger = logging.getLogger(
            logger_name
        )

        logger.addHandler(
            safe_handler
        )


    # --------------------------------------------------------
    # Start background tasks
    # --------------------------------------------------------

    t1 = asyncio.create_task(
        manager.auto_unload_task()
    )

    t2 = asyncio.create_task(
        heartbeat_status_task()
    )

    t3 = asyncio.create_task(
        sync_llama_log_task()
    )


    # --------------------------------------------------------
    # Start embedding server FIRST
    # --------------------------------------------------------

    ts_print(
        "🧠 Initializing embedding "
        "llama-server..."
    )


    embedding_ok = (
        await embedding_manager
        .ensure_started()
    )


    if embedding_ok:

        ts_print(
            "✅ Embedding backend ready"
        )

    else:

        ts_print(
            "⚠️ Embedding backend "
            "failed to start"
        )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    ts_print(
        f"📡 LLM upstream: "
        f"{UPSTREAM}"
    )

    ts_print(
        f"🧠 Embedding upstream: "
        f"{EMBEDDING_UPSTREAM}"
    )

    ts_print(
        f"📁 Local LLM log: "
        f"{LOCAL_LLAMA_LOG_FILE}"
    )

    ts_print(
        f"📁 Local embedding log: "
        f"{LOCAL_EMBED_LOG_FILE}"
    )


    update_status(
        "idle"
    )


    yield


    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    ts_print(
        "🛑 Gateway shutting down..."
    )


    update_status(
        "offline"
    )


    t1.cancel()
    t2.cancel()
    t3.cancel()


    manager.unload()

    embedding_manager.stop()


    if http_client is not None:

        await http_client.aclose()


# ============================================================
# APPLY LIFESPAN
# ============================================================

app.router.lifespan_context = lifespan


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    ts_print(
        "🔥 Starting Smart LLM Gateway"
    )

    ts_print(
        f"🌐 Gateway: "
        f"0.0.0.0:11434"
    )

    ts_print(
        f"🤖 LLM llama-server: "
        f"{UPSTREAM}"
    )

    ts_print(
        f"🧠 Embedding llama-server: "
        f"{EMBEDDING_UPSTREAM}"
    )


    config = Config(

        app=app,

        host="0.0.0.0",

        port=11434,

        loop="asyncio",

        lifespan="on",

        log_config=log_config

    )


    server = Server(
        config
    )


    try:

        asyncio.run(
            server.serve()
        )


    except OSError as e:

        if (
            "insufficient buffer space"
            in str(e)
            or
            "queue was full"
            in str(e)
        ):

            ts_print(
                "⚠️ Socket buffer issue. "
                "Waiting 30s..."
            )

            time.sleep(30)

            asyncio.run(
                server.serve()
            )

        else:

            raise

