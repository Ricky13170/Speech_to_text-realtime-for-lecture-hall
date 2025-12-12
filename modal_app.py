import modal
import asyncio
import json
import struct
import time
import numpy as np
from typing import Dict, Any

app = modal.App("asr-whisper-large-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "fastapi",
        "uvicorn",
        "websockets",
        "numpy",
        "deep-translator",
    )
)

SAMPLE_RATE = 16000
MAX_BUFFER_SEC = 10
MAX_BUFFER_SAMPLES = MAX_BUFFER_SEC * SAMPLE_RATE
SILENCE_LIMIT = 0.8
PARTIAL_INTERVAL = 0.4
MODEL_ID = "openai/whisper-large-v3"


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
)
class ASRService:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        print("Loading Whisper Large V3...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        from deep_translator import GoogleTranslator
        self.translator = GoogleTranslator(source='vi', target='en')
        print("Model loaded")

    def transcribe(self, audio_np):
        import torch

        if audio_np is None or len(audio_np) < int(0.5 * SAMPLE_RATE):
            return "", 0.0

        audio_duration = len(audio_np) / SAMPLE_RATE
        
        if audio_duration > 30.0:
            audio_np = audio_np[-int(30.0 * SAMPLE_RATE):]
            audio_duration = 30.0

        rms = np.sqrt(np.mean(audio_np**2))
        if rms < 0.01:
            return "", audio_duration

        try:
            inputs = self.processor(
                audio=audio_np,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

            with torch.no_grad():
                ids = self.model.generate(
                    input_features,
                    max_new_tokens=256,
                    language="vi",
                    task="transcribe",
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    num_beams=1,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            text = self.filter_hallucination(text, audio_duration)
            
            if text:
                print(f"[Transcribe] {text}")
            
            return text, audio_duration
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", audio_duration

    def filter_hallucination(self, text: str, audio_duration: float) -> str:
        if not text:
            return ""
        
        words = text.split()
        if not words:
            return ""
        
        if audio_duration < 0.5 and len(words) > 5:
            return ""
        
        max_words = int(audio_duration * 5)
        if len(words) > max_words:
            return ""
        
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return ""
        
        return text

    def translate(self, text: str) -> str:
        try:
            return self.translator.translate(text)
        except:
            return ""

    @modal.method()
    def process_audio(self, audio_bytes: bytes) -> dict:
        import numpy as np

        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text, duration = self.transcribe(pcm)
        trans = self.translate(text) if text else ""

        return {"text": text, "trans": trans, "duration": duration}


@app.function(
    image=image,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    import numpy as np

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    asr_service = ASRService()
    viewers: Dict[str, WebSocket] = {}

    class BroadcasterState:
        def __init__(self, ws: WebSocket):
            self.ws = ws
            self.buffer = np.zeros(0, dtype=np.float32)
            self.speech_active = False
            self.last_speech_ts = 0.0
            self.lock = asyncio.Lock()
            self.partial_task = None

    broadcasters: Dict[str, BroadcasterState] = {}

    async def broadcast_to_viewers(obj: dict):
        dead = []
        payload = json.dumps(obj)
        for cid, ws in list(viewers.items()):
            try:
                await ws.send_text(payload)
            except:
                dead.append(cid)
        for cid in dead:
            viewers.pop(cid, None)

    async def do_transcribe(state: BroadcasterState, final=False):
        async with state.lock:
            audio = state.buffer.copy()

        if audio.size == 0:
            return

        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: asr_service.process_audio.remote(audio_bytes)
        )

        if final:
            async with state.lock:
                state.buffer = np.zeros(0, dtype=np.float32)
            msg_type = "fullSentence"
        else:
            msg_type = "realtime"

        if result.get("text"):
            payload = {"type": msg_type, "text": result["text"], "trans": result["trans"]}
            print(f"[{msg_type}] {result['text']}")
            try:
                await state.ws.send_text(json.dumps(payload))
            except:
                pass
            await broadcast_to_viewers(payload)

    async def periodic_partial(state: BroadcasterState):
        try:
            while state.speech_active:
                await do_transcribe(state, final=False)
                await asyncio.sleep(PARTIAL_INTERVAL)
        except asyncio.CancelledError:
            pass

    @fastapi_app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        cid = str(id(ws))
        role = ws.query_params.get("role", "broadcaster")

        if role == "viewer":
            viewers[cid] = ws
            print(f"Viewer connected: {cid}")
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                viewers.pop(cid, None)
            return

        state = BroadcasterState(ws)
        broadcasters[cid] = state
        print(f"Broadcaster connected: {cid}")

        try:
            while True:
                data = await ws.receive_bytes()

                meta_len = struct.unpack_from("<I", data, 0)[0]
                audio_bytes = data[4 + meta_len:]

                if len(audio_bytes) == 0:
                    continue

                pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                async with state.lock:
                    state.buffer = np.concatenate([state.buffer, pcm])
                    if len(state.buffer) > MAX_BUFFER_SAMPLES:
                        state.buffer = state.buffer[-MAX_BUFFER_SAMPLES:]

                rms = np.sqrt(np.mean(pcm**2))
                is_speech = rms > 0.01
                now = time.time()

                if is_speech:
                    if not state.speech_active:
                        state.speech_active = True
                        state.last_speech_ts = now
                        await ws.send_text(json.dumps({"type": "vad_start"}))
                        await broadcast_to_viewers({"type": "vad_start"})
                        state.partial_task = asyncio.create_task(periodic_partial(state))
                    else:
                        state.last_speech_ts = now

                elif state.speech_active and (now - state.last_speech_ts) > SILENCE_LIMIT:
                    state.speech_active = False
                    if state.partial_task:
                        state.partial_task.cancel()
                        state.partial_task = None

                    await ws.send_text(json.dumps({"type": "vad_stop"}))
                    await broadcast_to_viewers({"type": "vad_stop"})
                    await do_transcribe(state, final=True)

        except WebSocketDisconnect:
            pass
        finally:
            if state.partial_task:
                state.partial_task.cancel()
            broadcasters.pop(cid, None)

    @fastapi_app.get("/")
    async def root():
        return {"status": "ok", "message": "Server is running", "websocket": "/ws"}

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}

    @fastapi_app.get("/viewer")
    async def viewer_page():
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ASR Live Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; text-align: center; }
        .status {
            padding: 10px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
        .status.connected { background: #d4edda; color: #155724; }
        .status.disconnected { background: #f8d7da; color: #721c24; }
        .status.speaking { background: #fff3cd; color: #856404; }
        .transcript-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            margin-bottom: 20px;
        }
        .partial {
            color: #666;
            font-style: italic;
            padding: 10px;
            background: #e9ecef;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .final-list { list-style: none; }
        .final-list li {
            padding: 15px;
            background: #fff;
            border-left: 4px solid #007bff;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .final-list .vi { font-size: 18px; color: #333; }
        .final-list .en { font-size: 14px; color: #666; margin-top: 5px; }
        .clear-btn {
            padding: 10px 20px;
            background: #dc3545;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .clear-btn:hover { background: #c82333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASR Live Transcription</h1>
        <div id="status" class="status disconnected">Disconnected</div>
        <div class="transcript-box">
            <div id="partial" class="partial" style="display:none;"></div>
            <ul id="finals" class="final-list"></ul>
        </div>
        <button class="clear-btn" onclick="clearTranscripts()">Clear</button>
    </div>
    <script>
        const wsUrl = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws?role=viewer';
        let ws;
        const statusEl = document.getElementById('status');
        const partialEl = document.getElementById('partial');
        const finalsEl = document.getElementById('finals');

        function connect() {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status connected';
            };
            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status disconnected';
                setTimeout(connect, 2000);
            };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'vad_start') {
                    statusEl.textContent = 'Speaking...';
                    statusEl.className = 'status speaking';
                } else if (data.type === 'vad_stop') {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'status connected';
                    partialEl.style.display = 'none';
                } else if (data.type === 'realtime') {
                    partialEl.textContent = data.text;
                    partialEl.style.display = 'block';
                } else if (data.type === 'fullSentence') {
                    partialEl.style.display = 'none';
                    const li = document.createElement('li');
                    li.innerHTML = '<div class="vi">' + data.text + '</div><div class="en">' + (data.trans || '') + '</div>';
                    finalsEl.insertBefore(li, finalsEl.firstChild);
                }
            };
        }

        function clearTranscripts() {
            finalsEl.innerHTML = '';
            partialEl.style.display = 'none';
        }

        connect();
    </script>
</body>
</html>
        """
        return HTMLResponse(html)

    return fastapi_app


@app.local_entrypoint()
def main():
    print("Deploy: modal deploy modal_app.py")
    print("Dev: modal serve modal_app.py")
