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
# ── Why this rewrite ────────────────────────────────────────────────────────────────────────────
# The previous gateway forwarded with a SYNCHRONOUS requests.Session inside async handlers, which
# blocks the single event loop for the ENTIRE llama-server generation. Consequences observed:
#   * while any generation runs, EVERY endpoint (even /v1/models, /health) goes unreachable;
#   * killing the client did NOT stop llama-server — the sync request kept the upstream generating,
#     so the gateway stayed blocked for minutes until it drained (needed a manual restart).
# This version forwards with httpx.AsyncClient so:
#   * the event loop is never blocked — mock/health/model endpoints answer instantly even mid-generation;
#   * a client disconnect (agent killed / step aborted) raises CancelledError, which closes the httpx
#     stream and CANCELS the upstream generation — no orphaned runaway that bricks the gateway.
# Everything else (on-demand load/unload, tier configs + auto-downgrade, Ollama mocks, API-key auth,
# reasoning_content recovery, sampling defaults) is preserved from the original.

import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from uvicorn.config import Config, LOGGING_CONFIG
from uvicorn.server import Server

log_config = LOGGING_CONFIG.copy()
log_config["formatters"]["access"]["fmt"] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
log_config["formatters"]["default"]["fmt"] = '[%(asctime)s] %(levelprefix)s %(message)s'
log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def ts_print(*args, **kwargs):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(now, *args, **kwargs)


load_dotenv()

app = FastAPI(title="Pentest LLM Gateway (async)")

API_KEY = os.getenv("API_KEY", "")

UPSTREAM = "http://localhost:8080"          # the backing llama-server.exe
# One shared async client. connect timeout is short (fast failure when llama-server is down); the read
# timeout is long because a slow local model legitimately generates for minutes — but because this is
# async it never blocks other requests while waiting.
http_client: httpx.AsyncClient | None = None
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=600.0)


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


class SmartLLMManager:
    def __init__(self):
        self.current_process = None
        self.current_model = None
        self.last_active_time = time.time()
        self.active_requests = 0
        self.lock = asyncio.Lock()

        # 🟢 Adjust your folder path here
        base_dir = "D:\\Program\\llama-b10054-bin-win-cuda-13.3-x64"
        exe = os.path.join(base_dir, "llama-server.exe")

        model_9b = os.path.join(base_dir, "model\\Qwen3.5-9B-GGUF\\Qwen3.5-9B-Q4_K_M.gguf")
        model_9b_uncen = os.path.join(base_dir, "model\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
        model_ornith = os.path.join(base_dir, "model\\Ornith-1.5-9B-uncensored\\Ornith-1.5-9B-uncensored.Q4_K_M.gguf")
        model_ornith_i1 = os.path.join(base_dir, "model\\Ornith-1.5-9B-uncensored-i1-GGUF\\Ornith-1.5-9B-uncensored.i1-Q5_K_M.gguf")

        model_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\Qwen3.5-4B-Q4_K_M.gguf")
        mmproj_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\mmproj-Qwen3.5-4B-BF16.gguf")

        # 🟢 Tier command repository (llama-server runs in the background on port 8080).
        # Prefix with & so PowerShell understands commands with quotes.
        self.configs = {
            "Ornith-1.5-9B-32k": f'& "{exe}" -m "{model_ornith}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-65k": f'& "{exe}" -m "{model_ornith}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-i1-32k": f'& "{exe}" -m "{model_ornith_i1}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Ornith-1.5-9B-i1-65k": f'& "{exe}" -m "{model_ornith_i1}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-32k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-65k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-132k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 131072 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-192k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 196608 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-32k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 32768 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-65k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-132k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 131072 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-9B-192k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 196608 -b 2048 -ub 1024 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
            "Qwen3.5-4B-64k": f'& "{exe}" -m "{model_4b}" --port 8080 -ngl 99 -c 65536 -b 2048 -ub 1024 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 8 -tb 8 --parallel 1 --cont-batching --jinja',
        }

    # Hard cap on how long to wait for a tier's /health after launch: an unbounded poll would hang a
    # load forever if a tier can't fit VRAM.
    LOAD_TIMEOUT_SECONDS = int(os.getenv("GATEWAY_LOAD_TIMEOUT", "180"))
    # Idle window before the model is unloaded to free VRAM (seconds).
    IDLE_UNLOAD_SECONDS = int(os.getenv("GATEWAY_IDLE_UNLOAD_SECONDS", "600"))

    async def _server_healthy(self) -> bool:
        """Quick async liveness probe of the backing llama-server on port 8080."""
        try:
            r = await http_client.get(f"{UPSTREAM}/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _downgrade_tier(self, model_name: str):
        """The next-smaller-context tier of the SAME model family (one-way auto-downgrade on VRAM
        pressure), or None if already the smallest. Preserves the variant suffix; never upgrades."""
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
            # Normalize model name (strip litellm openai/ prefix, :latest suffix, whitespace).
            clean_name = (model_name or "").replace("openai/", "").replace(":latest", "").strip()
            if clean_name in self.configs:
                model_name = clean_name
            else:
                case_map = {k.lower(): k for k in self.configs}
                if clean_name.lower() in case_map:
                    model_name = case_map[clean_name.lower()]
                else:
                    ts_print(f"⚠️ Model '{model_name}' not in the known tier list "
                             f"({', '.join(self.configs)}) -> defaulting to 'Ornith-1.5-9B-65k'")
                    model_name = "Ornith-1.5-9B-65k"

            if self.current_model == model_name and self.current_process is not None:
                proc_dead = self.current_process.poll() is not None
                if proc_dead or not await self._server_healthy():
                    ts_print("⚠️ [Smart Gateway] Backing llama-server is not responding "
                             f"(proc_dead={proc_dead}) -> reloading [{model_name}]...")
                    self.unload()
                else:
                    self.last_active_time = time.time()
                    return

            self.unload()
            while True:
                ts_print(f"\n🚀 [Smart Gateway] Loading [{model_name}] into VRAM...")
                await asyncio.sleep(2)
                cmd = self.configs[model_name]
                self.current_process = subprocess.Popen(
                    ["powershell", "-Command", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.current_model = model_name
                self.last_active_time = time.time()

                deadline = time.time() + self.LOAD_TIMEOUT_SECONDS
                ready = False
                while time.time() < deadline:
                    if self.current_process.poll() is not None:
                        break                        # child exited (OOM) → stop polling, downgrade
                    if await self._server_healthy():
                        ready = True
                        break
                    await asyncio.sleep(1)
                if ready:
                    ts_print(f"✅ [Smart Gateway] Successfully loaded [{model_name}]! Ready to forward to agent")
                    break
                # Not healthy within the hard timeout (or the process died) → one-way downgrade.
                self.unload()
                smaller = self._downgrade_tier(model_name)
                if smaller and smaller in self.configs:
                    ts_print(f"⚠️ [Smart Gateway] [{model_name}] failed to load within "
                             f"{self.LOAD_TIMEOUT_SECONDS}s (likely VRAM OOM). Auto-downgrading → [{smaller}].")
                    model_name = smaller
                    continue
                ts_print(f"❌ [Smart Gateway] [{model_name}] failed to load and no smaller tier "
                         f"is available — giving up (likely VRAM). The agent will see the gateway as down.")
                self.current_model = None
                break

    def unload(self):
        if self.current_process:
            ts_print("💤 [Smart Gateway] Clearing GPU VRAM (Force Kill llama-server)...")
            subprocess.run(
                ["powershell", "-Command", "Stop-Process -Name 'llama-server' -Force -ErrorAction SilentlyContinue"],
                check=False,
            )
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=1)
            except Exception:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
            self.current_process = None
            self.current_model = None

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


@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    asyncio.create_task(manager.auto_unload_task())


@app.on_event("shutdown")
async def shutdown_event():
    if http_client is not None:
        await http_client.aclose()


# ── Ollama mock endpoints (served BY the gateway itself → always instant, never forwarded) ────────
@app.get("/api/tags")
async def ollama_tags():
    models = []
    for name in manager.configs:
        models.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "size": 4000000000,
            "digest": "sha256:" + "0" * 64,
            "details": {"format": "gguf", "family": "llama",
                        "parameter_size": "4B" if "4B" in name else "9B",
                        "quantization_level": "Q4_0"},
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
    # Gateway-level liveness that does NOT require the model — always answers so callers can tell the
    # gateway is up even while a model is loading or a generation is in flight.
    return {"status": "ok", "model": manager.current_model, "active_requests": manager.active_requests}


@app.post("/api/show")
async def ollama_show(request: Request):
    try:
        body = await request.json()
        model_name = body.get("name", "Ornith-1.5-9B-65k").replace(":latest", "")
    except Exception:
        model_name = "Ornith-1.5-9B-65k"
    cmd = manager.configs.get(model_name, "")
    ctx = _ctx_of(cmd)
    return {
        "modelfile": f"FROM {model_name}\nPARAMETER num_ctx {ctx}\n",
        "parameters": f"num_ctx                        {ctx}\n",
        "template": "{{ .Prompt }}",
        "details": {"format": "gguf", "family": "llama",
                    "parameter_size": "4B" if "4B" in model_name else "9B",
                    "quantization_level": "Q4_0"},
        "model_info": {"llama.context_length": ctx, "qwen2.context_length": ctx,
                       "general.context_length": ctx},
    }


def _recover_content_from_reasoning(resp_json: dict) -> dict:
    """If a thinking model emitted everything in reasoning_content and left content empty, recover the
    JSON block (or the reasoning text) into content so standard OpenAI clients get a valid answer."""
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


# ── API intercept & forward (async, non-blocking, cancellable) ───────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_llm(path: str, request: Request):
    body = await request.body()

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

    # 1. Ensure the model is loaded (on-demand). active_requests is incremented for the whole lifetime
    #    of this request and decremented in a finally, so the idle-unload never fires mid-request.
    counted = False
    if is_chat and request.method == "POST":
        manager.active_requests += 1
        counted = True
        requested_model = req_data.get("model", "Ornith-1.5-9B-65k") if isinstance(req_data, dict) else "Ornith-1.5-9B-65k"
        try:
            await manager.ensure_model(requested_model)
        except Exception as e:
            ts_print("Error ensuring model:", e)
            await manager.ensure_model("Ornith-1.5-9B-65k")

    def _release():
        nonlocal counted
        if counted:
            manager.active_requests -= 1
            manager.last_active_time = time.time()
            counted = False

    # 2. Rewrite the body for llama-server: force model name to "default", apply Ornith/Qwen default
    #    sampling if the caller did not set it, and preserve every other field (reasoning,
    #    chat_template_kwargs, etc.) verbatim.
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

    # 3a. NON-STREAMING — a single awaited forward. Because it is async, the event loop stays free to
    #     serve health/model endpoints while llama-server generates. A client disconnect cancels this
    #     coroutine, and httpx then cancels the upstream request (llama-server stops generating).
    if not is_stream:
        try:
            r = await http_client.request(
                request.method, url, headers=headers, content=body, params=request.query_params,
            )
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            _release()
            if isinstance(e, asyncio.CancelledError):
                raise
            ts_print(f"Connection to llama-server failed: {e}")
            return Response(content=json.dumps({"error": f"llama-server unreachable: {e}"}),
                            status_code=502, media_type="application/json")
        try:
            if r.status_code != 200:
                return Response(content=r.content, status_code=r.status_code,
                                media_type=r.headers.get("content-type", "application/json"))
            if is_ollama:
                resp_json = r.json()
                choices = resp_json.get("choices") or [{}]
                msg = (choices[0] if choices else {}).get("message") or {}
                content = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
                return JSONResponse(content={
                    "model": manager.current_model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                })
            resp_json = _recover_content_from_reasoning(r.json())
            return JSONResponse(content=resp_json, status_code=r.status_code)
        except Exception:
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
        finally:
            _release()

    # 3b. STREAMING — stream chunks as they arrive. A client disconnect raises CancelledError inside the
    #     generator; the `async with client.stream()` context then closes the upstream connection, which
    #     cancels llama-server's generation. active_requests is released in the finally either way.
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
        except asyncio.CancelledError:
            ts_print("client disconnected → cancelling upstream generation")
            raise
        except httpx.HTTPError as e:
            ts_print(f"Proxy streaming error: {e}")
            yield json.dumps({"error": "proxy failed"}).encode() + b"\n"
        finally:
            _release()

    return StreamingResponse(
        generate(), status_code=200,
        media_type="text/event-stream" if "v1/" in path else "application/x-ndjson",
    )


if __name__ == "__main__":
    ts_print("🔥 Starting Pentest Smart Gateway (async) on port 11434 (mocking Ollama)...")
    config = Config(app=app, host="0.0.0.0", port=11434, loop="asyncio",
                    lifespan="on", log_config=log_config)
    server = Server(config)
    try:
        asyncio.run(server.serve())
    except OSError as e:
        if "insufficient buffer space" in str(e) or "queue was full" in str(e):
            ts_print("⚠️  Socket buffer issue. Waiting 30s and retrying...")
            time.sleep(30)
            asyncio.run(server.serve())
        else:
            raise
