import asyncio
import json
import struct
import time
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from deep_translator import GoogleTranslator

import config
from audio_processor import vad_prob, process_realtime, process_final, local_agreement

processor = None
model = None
translator = GoogleTranslator(source='vi', target='en')
caption_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, caption_process
    print(f"Loading model on {config.DEVICE}...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(config.MODEL_ID).to(config.DEVICE)
    model.eval()
    print("Model ready")

    if config.USE_CAPTION_OVERLAY:
        caption_process = subprocess.Popen([sys.executable, "caption.py"])

    yield

    if caption_process:
        caption_process.terminate()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

broadcasters: Dict[str, "ConnectionState"] = {}
viewers: Dict[str, WebSocket] = {}


class ConnectionState:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.buffer = np.zeros(0, dtype=np.float32)
        self.partial_history = []
        self.agreed_text = ""
        self.speech_active = False
        self.last_speech_ts = 0.0
        self.lock = asyncio.Lock()
        self.partial_task = None

    async def send(self, obj: Dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception:
            pass


async def broadcast(obj: Dict[str, Any]):
    if not viewers:
        return
    payload = json.dumps(obj)
    dead = []
    for cid, ws in list(viewers.items()):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(cid)
    for cid in dead:
        viewers.pop(cid, None)


def transcribe(audio_np):
    if audio_np is None or len(audio_np) < int(0.5 * config.SAMPLE_RATE):
        return ""
    try:
        inputs = processor(audio=audio_np, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(config.DEVICE)

        with torch.no_grad():
            ids = model.generate(
                features,
                max_new_tokens=128,
                language="vi",
                task="transcribe",
                pad_token_id=processor.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"Transcribe error: {e}")
        return ""


async def do_transcribe(state: ConnectionState, final=False):
    async with state.lock:
        audio = state.buffer.copy()

    if audio.size == 0:
        return

    audio_proc = process_final(audio) if final else process_realtime(audio)

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, lambda: transcribe(audio_proc))
    except asyncio.CancelledError:
        return

    if not text:
        return

    try:
        trans = await loop.run_in_executor(None, lambda: translator.translate(text))
    except Exception:
        trans = ""

    if final:
        payload = {"type": "fullSentence", "text": text, "trans": trans}
        await state.send(payload)
        await broadcast(payload)
        async with state.lock:
            state.buffer = np.zeros(0, dtype=np.float32)
            state.partial_history = []
            state.agreed_text = ""
    else:
        async with state.lock:
            state.partial_history.append(text)
            if len(state.partial_history) > 5:
                state.partial_history = state.partial_history[-5:]
            agreed = local_agreement(state.partial_history)
            if agreed and len(agreed) > len(state.agreed_text):
                state.agreed_text = agreed

        display = state.agreed_text or text
        payload = {"type": "realtime", "text": display, "trans": trans}
        await state.send(payload)
        await broadcast(payload)


async def partial_sender(state: ConnectionState):
    try:
        while state.speech_active:
            await do_transcribe(state, final=False)
            await asyncio.sleep(config.PARTIAL_INTERVAL)
    except asyncio.CancelledError:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = str(id(ws))
    role = ws.query_params.get("role", "broadcaster")

    if role == "viewer":
        viewers[cid] = ws
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            viewers.pop(cid, None)
        return

    state = ConnectionState(ws)
    broadcasters[cid] = state

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
                if len(state.buffer) > config.MAX_BUFFER_SAMPLES:
                    state.buffer = state.buffer[-config.MAX_BUFFER_SAMPLES:]

            prob = vad_prob(pcm)
            now = time.time()

            if prob >= config.VAD_THRESHOLD:
                if not state.speech_active:
                    state.speech_active = True
                    state.last_speech_ts = now
                    await state.send({"type": "vad_start"})
                    await broadcast({"type": "vad_start"})
                    state.partial_task = asyncio.create_task(partial_sender(state))
                else:
                    state.last_speech_ts = now

            elif state.speech_active and (now - state.last_speech_ts) > config.SILENCE_LIMIT:
                state.speech_active = False
                if state.partial_task:
                    state.partial_task.cancel()
                    state.partial_task = None
                await state.send({"type": "vad_stop"})
                await broadcast({"type": "vad_stop"})
                await do_transcribe(state, final=True)

    except WebSocketDisconnect:
        pass
    finally:
        if state.partial_task:
            state.partial_task.cancel()
        broadcasters.pop(cid, None)


@app.get("/")
async def serve_viewer():
    with open("overlay_client.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, log_level="info")
