# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "requests",
#     "uvicorn",
#     "python-dotenv",
# ]
# ///

import subprocess
import time
import requests
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import asyncio
import socket
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

app = FastAPI(title="Pentest 4-Tier LLM Gateway")

API_KEY = os.getenv("API_KEY", "")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
        
    # Safe services (status check / model list) are allowed without a Key
    safe_paths = [
        "/api/tags",
        "/v1/models",
        "/api/v1/models",
        "/api/version",
        "/version",
        "/v1/props",
        "/props",
        "/api/show"
    ]
    if request.url.path in safe_paths:
        return await call_next(request)
        
    # ตรวจสอบ API Key เฉพาะกรณีที่มีการตั้งค่าไว้เท่านั้น
    if API_KEY:
        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"error": "Unauthorized: Invalid or missing API Key"})
        
    return await call_next(request)

session = requests.Session()

class SmartLLMManager:
    def __init__(self):
        self.current_process = None
        self.current_model = None
        self.last_active_time = time.time()
        self.active_requests = 0
        
        # 🟢 Adjust your folder path here
        base_dir = "D:\\Program\\llama-b10054-bin-win-cuda-13.3-x64"
        exe = os.path.join(base_dir, "llama-server.exe")
        
        model_9b = os.path.join(base_dir, "model\\Qwen3.5-9B-GGUF\\Qwen3.5-9B-Q4_K_M.gguf")
        model_9b_uncen = os.path.join(base_dir, "model\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
        model_9b_claude = os.path.join(base_dir, "model\\Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED-THINKING-MAX-NEOCODE-Imatrix-GGUF\\Qwen3.5-9B-Claude-4.6-OS-AV-H-UNCENSORED-THINK-D_AU-Q4_K_S-imat.gguf")
        model_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\Qwen3.5-4B-Q4_K_M.gguf") # Adjust to match the actual name
        mmproj_4b = os.path.join(base_dir, "model\\Qwen3.5-4B-GGUF\\mmproj-Qwen3.5-4B-BF16.gguf")
        
        # 🟢 Tier command repository (running in the background on port 8080!)
        # Prefix with & so PowerShell understands commands with quotes
        self.configs = {
            "analyst-32k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 32768 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-65k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 65536 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-132k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 131072 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-192k-uncen": f'& "{exe}" -m "{model_9b_uncen}" --port 8080 -ngl 99 -c 196608 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',

            "analyst-32k-claude": f'& "{exe}" -m "{model_9b_claude}" --port 8080 -ngl 99 -c 32768 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-65k-claude": f'& "{exe}" -m "{model_9b_claude}" --port 8080 -ngl 99 -c 65536 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-132k-claude": f'& "{exe}" -m "{model_9b_claude}" --port 8080 -ngl 99 -c 131072 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-192k-claude": f'& "{exe}" -m "{model_9b_claude}" --port 8080 -ngl 99 -c 196608 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',

            "analyst-32k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 32768 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-65k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 65536 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-132k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 131072 --no-mmap -fa on -ctk q8_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',
            "analyst-192k": f'& "{exe}" -m "{model_9b}" --port 8080 -ngl 99 -c 196608 --no-mmap -fa on -ctk q4_0 -ctv q4_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja',

            # 4B text model  (lighter/faster than the 9B tiers)
            "omni-vision-64k": f'& "{exe}" -m "{model_4b}" --port 8080 -ngl 99 -c 65536 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4 -tb 8 --parallel 1 --cont-batching --jinja'
        }

    # Hard cap on how long to wait for a tier's /health after launch (Gateway §5):
    # an unbounded poll hangs the whole run if a tier can't fit VRAM.
    LOAD_TIMEOUT_SECONDS = int(os.getenv("GATEWAY_LOAD_TIMEOUT", "180"))

    def _server_healthy(self) -> bool:
        """Quick liveness probe of the backing llama-server on port 8080."""
        try:
            return session.get("http://localhost:8080/health", timeout=2).status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _downgrade_tier(self, model_name: str):
        """The next-smaller-context tier of the SAME model family (Gateway §5 one-way
        auto-downgrade on VRAM pressure), or None if already the smallest. Preserves the
        variant suffix (``''`` / ``-claude`` / ``-uncen``); never upgrades."""
        m = re.match(r"analyst-(\d+)k(-claude|-uncen)?$", model_name or "")
        if not m:
            return None
        ctx, suffix = int(m.group(1)), (m.group(2) or "")
        smaller = sorted((int(re.match(r"analyst-(\d+)k", k).group(1))
                          for k in self.configs
                          if re.match(rf"analyst-\d+k{re.escape(suffix)}$", k)),
                         reverse=True)
        nxt = next((c for c in smaller if c < ctx), None)
        return f"analyst-{nxt}k{suffix}" if nxt is not None else None

    async def ensure_model(self, model_name):
        if not hasattr(self, 'lock'):
            self.lock = asyncio.Lock()

        async with self.lock:
            # If no model name is specified, or an unknown name is provided, use the default: analyst-65k
            if model_name not in self.configs:
                ts_print(f"⚠️ Model '{model_name}' is not in the known tier list ({', '.join(self.configs)}) "
                      f"-> Defaulting to 'analyst-65k'")
                model_name = "analyst-65k"

            if self.current_model == model_name and self.current_process is not None:
                # Auto-recovery: the requested model is nominally loaded, but the
                # llama-server child may have died (e.g. VRAM OOM). If the process
                # exited or /health is unresponsive, force a reload instead of
                # forwarding to a dead backend (which surfaced to the agent as
                # "Cannot connect to Smart Gateway" and stalled the whole run).
                proc_dead = self.current_process.poll() is not None
                if proc_dead or not self._server_healthy():
                    ts_print("⚠️ [Smart Gateway] Backing llama-server is not responding "
                          f"(proc_dead={proc_dead}) -> reloading [{model_name}]...")
                    self.unload()
                else:
                    self.last_active_time = time.time()
                    return

            self.unload()
            # Launch the requested tier; if it does not become healthy within the HARD
            # timeout (almost always a VRAM OOM for a too-large context), auto-downgrade
            # ONE step to the next-smaller-context tier of the same family and retry —
            # a bounded, one-way degrade so the run continues DEGRADED instead of hanging
            # forever on a dead backend (Gateway §5).
            while True:
                ts_print(f"\n🚀 [Smart Gateway] Loading [{model_name}] into VRAM...")
                # Delay 2s so Windows fully frees VRAM from the previous instance
                # (prevents RAM spikes that act like two instances at once).
                await asyncio.sleep(2)
                cmd = self.configs[model_name]
                # Run directly via PowerShell instead of shell=True (which is cmd.exe).
                self.current_process = subprocess.Popen(["powershell", "-Command", cmd],
                                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_model = model_name
                self.last_active_time = time.time()

                deadline = time.time() + self.LOAD_TIMEOUT_SECONDS
                ready = False
                while time.time() < deadline:
                    if self.current_process.poll() is not None:
                        break                       # child exited (OOM) → stop polling, downgrade
                    try:
                        if session.get("http://localhost:8080/health", timeout=1).status_code == 200:
                            ready = True
                            break
                    except requests.exceptions.RequestException:
                        pass
                    await asyncio.sleep(1)
                if ready:
                    ts_print(f"✅ [Smart Gateway] Successfully loaded [{model_name}]! Ready to forward data to Hermes Agent")
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
            ts_print("💤 [Smart Gateway] Clearing GPU VRAM to prepare for loading a new model (Force Kill)...")
            # Kill the llama-server process directly to immediately free up VRAM (similar to closing the script)
            subprocess.run(["powershell", "-Command", "Stop-Process -Name 'llama-server' -Force -ErrorAction SilentlyContinue"], check=False)
            
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=1)
            except:
                self.current_process.kill()
            self.current_process = None
            self.current_model = None

    # Idle window before the model is unloaded to free VRAM (seconds).
    IDLE_UNLOAD_SECONDS = int(os.getenv("GATEWAY_IDLE_UNLOAD_SECONDS", "600"))  # 10 minutes

    async def auto_unload_task(self):
        idle = self.IDLE_UNLOAD_SECONDS
        mins = idle // 60
        while True:
            await asyncio.sleep(30) # Check every 30 seconds
            if self.current_process and self.active_requests == 0 and (time.time() - self.last_active_time > idle):
                if hasattr(self, 'lock'):
                    async with self.lock:
                        if self.current_process and self.active_requests == 0 and (time.time() - self.last_active_time > idle):
                            ts_print(f"⏳ [Smart Gateway] No model usage for over {mins} minutes -> Auto-closing model to save VRAM")
                            self.unload()
                else:
                    ts_print(f"⏳ [Smart Gateway] No model usage for over {mins} minutes -> Auto-closing model to save VRAM")
                    self.unload()

manager = SmartLLMManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(manager.auto_unload_task())


# --- Ollama Mock Endpoints (Act exactly like Ollama) ---
@app.get("/api/tags")
async def ollama_tags():
    models = []
    for name, cmd in manager.configs.items():
        models.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "size": 4000000000,
            "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "details": {
                "format": "gguf", 
                "family": "llama", 
                "parameter_size": "4B" if "scout" in name else "9B", 
                "quantization_level": "Q4_0"
            }
        })
    return {"models": models}

@app.get("/v1/models")
@app.get("/api/v1/models")
async def openai_models():
    data = []
    for name, cmd in manager.configs.items():
        ctx_size = 8192
        if "-c " in cmd:
            try:
                ctx_size = int(cmd.split("-c ")[1].split(" ")[0])
            except:
                pass
        data.append({
            "id": name, 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "library",
            "max_model_len": ctx_size,
            "context_window": ctx_size
        })
    return {"object": "list", "data": data}

@app.get("/api/version")
@app.get("/version")
async def ollama_version():
    return {"version": "0.1.48"}

@app.get("/v1/props")
async def v1_props():
    return {"properties": {}}

@app.post("/api/show")
async def ollama_show(request: Request):
    try:
        body = await request.json()
        model_name = body.get("name", "analyst-65k").replace(":latest", "")
    except:
        model_name = "analyst-65k"
        
    cmd = manager.configs.get(model_name, "")
    ctx_size = 8192
    if "-c " in cmd:
        try:
            ctx_size = int(cmd.split("-c ")[1].split(" ")[0])
        except:
            pass
            
    modelfile = f"FROM {model_name}\nPARAMETER num_ctx {ctx_size}\n"
    parameters = f"num_ctx                        {ctx_size}\n"
    
    return {
        "modelfile": modelfile,
        "parameters": parameters,
        "template": "{{ .Prompt }}",
        "details": {
            "format": "gguf", 
            "family": "llama", 
            "parameter_size": "4B" if "scout" in model_name else "9B", 
            "quantization_level": "Q4_0"
        },
        "model_info": {
            "llama.context_length": ctx_size,
            "qwen2.context_length": ctx_size,
            "general.context_length": ctx_size
        }
    }

# --- API Intercept & Forward System ---
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_llm(path: str, request: Request):
    body = await request.body()
    
    # 1. Check and load the model
    is_chat = "chat/completions" in path or "api/chat" in path or "api/generate" in path or "v1/completions" in path
    if is_chat and request.method == "POST":
        manager.active_requests += 1
        try:
            json_data = await request.json()
            requested_model = json_data.get("model", "analyst-65k").replace(":latest", "")
            await manager.ensure_model(requested_model)
        except Exception as e:
            ts_print("Error parsing json:", e)
            await manager.ensure_model("analyst-65k")


    # 2. Convert Ollama Request format to OpenAI for llama-server.exe and correct the model name
    target_path = path
    if request.method in ["POST", "PUT"]:
        try:
            # Force the model name to "default" so llama-server.exe does not reject the Request
            req_data = await request.json()
            if "model" in req_data:
                req_data["model"] = "default"

                # Apply Qwen official default sampling parameters if not explicitly provided
                req_data.setdefault("top_p", 0.8)
                req_data.setdefault("top_k", 20)
                req_data.setdefault("min_p", 0.0)
                req_data.setdefault("presence_penalty", 0.0)
                req_data.setdefault("repetition_penalty", 1.0)

                body = json.dumps(req_data).encode("utf-8")
        except:
            pass

    if path == "api/chat":
        target_path = "v1/chat/completions"
    elif path == "api/generate":
        target_path = "v1/completions"

    # 3. Forward Request to llama-server (port 8080)
    url = f"http://localhost:8080/{target_path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() not in ['host', 'content-length']}
    
    try:
        # Send the Request and wait to receive Headers and Status Code first
        resp = session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body,
            params=request.query_params,
            stream=True,
            timeout=600
        )
    except Exception as e:
        ts_print(f"Connection to llama-server failed: {e}")
        return Response(content=f'{{"error": "Connection to llama-server failed: {e}"}}', status_code=502)

    # If llama-server returns an Error (e.g. 400 Bad Request), return it immediately
    if resp.status_code != 200:
        content = resp.content
        resp.close()
        return Response(content=content, status_code=resp.status_code, headers=dict(resp.headers))

    # 4. Use StreamingResponse and convert the Response back to Ollama Format
    def generate():
        try:
            is_ollama = (path in ["api/chat", "api/generate"])
            if not is_ollama:
                # For OpenAI Endpoint (/v1/...), let the raw Stream pass through unmodified (prevents empty line bugs)
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            else:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    
                    # Convert OpenAI format (data: {...}) to Ollama JSON
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        json_str = decoded[6:]
                        if json_str.strip() == '[DONE]':
                            yield (json.dumps({"model": manager.current_model, "done": True}) + "\n").encode('utf-8')
                            continue
                        try:
                            data = json.loads(json_str)
                            if "api/chat" in path:
                                content = data['choices'][0]['delta'].get('content', '')
                                ollama_chunk = {
                                    "model": data.get("model", manager.current_model),
                                    "message": {"role": "assistant", "content": content},
                                    "done": False
                                }
                            else:
                                content = data['choices'][0].get('text', '')
                                ollama_chunk = {
                                    "model": data.get("model", manager.current_model),
                                    "response": content,
                                    "done": False
                                }
                            yield (json.dumps(ollama_chunk) + "\n").encode('utf-8')
                        except Exception as e:
                            pass
        except Exception as e:
            ts_print(f"Proxy Streaming Error: {e}")
            yield b'{"error": "proxy failed"}\n'
        finally:
            resp.close()
            if is_chat and request.method == "POST":
                manager.active_requests -= 1
                manager.last_active_time = time.time()
            
    return StreamingResponse(generate(), status_code=resp.status_code, media_type="text/event-stream" if "v1/" in path else "application/x-ndjson")

if __name__ == "__main__":
    ts_print("🔥 Starting Pentest Smart Gateway on port 11434 (Mocking Ollama)...")

    config = Config(
        app=app,
        host="0.0.0.0",
        port=11434,
        loop="asyncio",
        lifespan="off",
        log_config=log_config,
    )
    server = Server(config)

    try:
        asyncio.run(server.serve())
    except OSError as e:
        if "insufficient buffer space" in str(e) or "queue was full" in str(e):
            ts_print("⚠️  Socket buffer issue detected. Waiting 30 seconds and retrying...")
            time.sleep(30)
            asyncio.run(server.serve())
        else:
            raise