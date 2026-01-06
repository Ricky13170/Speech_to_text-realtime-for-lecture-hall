import sys
import pathlib
from modal import App, Image, asgi_app, Volume, enter

sys.path.append(str(pathlib.Path(__file__).parent))

from backend.config import (
    MODAL_APP_NAME, MODAL_GPU, MODAL_MEMORY, MODAL_TIMEOUT,
    MODAL_CONTAINER_IDLE_TIMEOUT, WHISPER_MODEL
)

app = App(MODAL_APP_NAME)

# Định nghĩa môi trường chạy (Container Image)
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1") 
    .pip_install(
        "torch", "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers>=4.35.0",
        "accelerate",
        "fastapi", "uvicorn", "websockets",
        "numpy", "scipy", "soundfile",
        "aiofiles",
        "safetensors",
        "protobuf"
    )
    # Copy code vào container
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_dir("frontend", remote_path="/root/frontend")
)

# Tạo Volume để cache model (tránh download lại mỗi lần khởi động)
cache = Volume.from_name("asr-model-cache", create_if_missing=True)

@app.cls(
    gpu=MODAL_GPU,
    memory=MODAL_MEMORY,
    timeout=MODAL_TIMEOUT,
    scaledown_window=MODAL_CONTAINER_IDLE_TIMEOUT,
    image=image,
    volumes={"/cache": cache}
)
class ASR:
    @enter()
    def setup(self):
        """Hàm này chạy 1 lần khi Container khởi động (Cold Start)"""
        import sys
        import os
        import time
        
        # Setup đường dẫn và cache
        sys.path.append("/root")
        os.environ["HF_HOME"] = "/cache/huggingface"
        
        print("[Container] Initializing models...")
        start = time.time()
        
        # Load toàn bộ Model vào GPU ngay lập tức
        from backend.server.websocket_handler import WebSocketHandler
        self.handler = WebSocketHandler()
        
        import asyncio
        asyncio.run(self.handler.init())
        
        print(f"[Container] Ready in {time.time() - start:.2f}s! Model: {WHISPER_MODEL}")

    @asgi_app()
    def web(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import JSONResponse
        import time
        import traceback
        
        web_app = FastAPI(
            title="ASR Thesis API",
            version="1.0.0",
            description="Vietnamese-English Real-time Speech Translation"
        )
        
        # 1. CORS
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 2. MIDDLEWARE CHỐNG CACHE (FIX LỖI DÍNH INDEX CŨ)
        @web_app.middleware("http")
        async def add_no_cache_header(request: Request, call_next):
            response = await call_next(request)
            # Yêu cầu trình duyệt không bao giờ lưu cache
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        
        start_time = time.time()
        metrics = {"ws_connections": 0, "transcriptions": 0, "errors": 0}
        
        # --- API ENDPOINTS ---
        @web_app.get("/")
        async def root():
            return JSONResponse({
                "status": "online",
                "service": "ASR Thesis Backend",
                "uptime": int(time.time() - start_time),
                "tip": "Go to /app/ for the interface"
            })

        @web_app.get("/health")
        async def health():
            return {"status": "healthy", "uptime": int(time.time() - start_time)}
            
        @web_app.websocket("/ws/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            metrics["ws_connections"] += 1
            client_id = metrics["ws_connections"]
            
            await websocket.accept()
            print(f"[WS #{client_id}] Connected")
            
            try:
                await websocket.send_json({"type": "status", "status": "connected"})
                while True:
                    message = await websocket.receive_text()
                    response = await self.handler.handle(message)
                    if response:
                        await websocket.send_text(response)
                        metrics["transcriptions"] += 1
            except WebSocketDisconnect:
                print(f"[WS #{client_id}] Disconnected")
            except Exception as e:
                print(f"[WS #{client_id}] Error: {e}")
                traceback.print_exc()

        # --- STATIC FILES ---
        web_app.mount("/app", StaticFiles(directory="/root/frontend", html=True), name="frontend")
        
        return web_app
