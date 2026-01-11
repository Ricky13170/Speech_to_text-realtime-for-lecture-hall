import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent))

from modal import App, Image, asgi_app, Volume, enter
from backend.config import MODAL_APP_NAME, MODAL_GPU, MODAL_MEMORY, MODAL_TIMEOUT, MODAL_CONTAINER_IDLE_TIMEOUT, WHISPER_MODEL

app = App(MODAL_APP_NAME)

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
    )
    .pip_install("openai-whisper>=20231117")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "aiofiles>=23.0.0",
    )
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.0",
    )
    .pip_install("safetensors>=0.4.0", "protobuf>=3.20.0")
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_dir("frontend", remote_path="/root/frontend")
)

cache = Volume.from_name("asr-model-cache", create_if_missing=True)


@app.cls(
    gpu=MODAL_GPU,
    memory=MODAL_MEMORY,
    timeout=MODAL_TIMEOUT,
    scaledown_window=MODAL_CONTAINER_IDLE_TIMEOUT,
    image=image,
    volumes={"/cache": cache},
)
class ASR:
    @enter()
    def setup(self):
        import sys
        import os
        import time
        import asyncio
        
        sys.path.append("/root")
        os.environ["HF_HOME"] = "/cache/huggingface"
        
        print("[Container] Initializing...")
        start = time.time()
        
        from backend.handler import ASRService
        self.service = ASRService()
        asyncio.run(self.service.init())
        
        print(f"[Container] Ready in {time.time() - start:.1f}s | Whisper: {WHISPER_MODEL} | GPU: {MODAL_GPU}")

    @asgi_app()
    def app(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import JSONResponse, FileResponse
        from starlette.middleware.base import BaseHTTPMiddleware
        import time
        import asyncio

        web_app = FastAPI(title="ASR", version="5.0")
        
        # No-cache middleware
        class NoCacheMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                response = await call_next(request)
                if request.url.path.endswith(('.js', '.css', '.html')):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"
                return response
        
        web_app.add_middleware(NoCacheMiddleware)
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        start_time = time.time()
        conn_count = [0]

        @web_app.get("/api/status")
        async def status():
            return JSONResponse({
                "status": "online",
                "model": WHISPER_MODEL,
                "gpu": MODAL_GPU,
                "uptime": int(time.time() - start_time),
            })

        @web_app.websocket("/ws/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            conn_count[0] += 1
            cid = conn_count[0]
            
            await websocket.accept()
            print(f"[WS #{cid}] Connected")
            
            session = self.service.create_session()
            
            async def recv():
                try:
                    while True:
                        msg = await websocket.receive_text()
                        await session.handle_incoming(msg)
                except WebSocketDisconnect:
                    print(f"[WS #{cid}] Disconnected")
                except Exception as e:
                    print(f"[WS #{cid}] Error: {e}")

            async def send():
                try:
                    while True:
                        msg = await session.out_queue.get()
                        await websocket.send_text(msg)
                except Exception:
                    pass

            try:
                done, pending = await asyncio.wait(
                    [asyncio.create_task(recv()), asyncio.create_task(send())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
            except Exception:
                pass
            finally:
                print(f"[WS #{cid}] Closed")

        web_app.mount("/", StaticFiles(directory="/root/frontend", html=True), name="frontend")
        return web_app