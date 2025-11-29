import asyncio
import json
import struct
import time
import os
import subprocess
import sys

# Tăng timeout cho Hugging Face Hub
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from deep_translator import GoogleTranslator

import config
from audio_processor import vad_prob_for_buffer, process_realtime_chunk, process_final_sentence

# ----------------- GLOBAL VARIABLES -----------------
processor = None
model = None
translator = GoogleTranslator(source='vi', target='en')
caption_process = None  # Biến quản lý tiến trình Caption

# ----------------- LIFESPAN (Load Model Once) -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, caption_process
    print(f"Loading PhoWhisper on {config.DEVICE}...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(config.MODEL_ID).to(config.DEVICE)
    model.eval()
    print("PhoWhisper ready")
    
    # Start Caption App (Subprocess)
    if config.USE_CAPTION_OVERLAY:
        print("📷 Launching Caption UI...")
        # Chạy caption.py như một chương trình riêng biệt
        caption_process = subprocess.Popen([sys.executable, "caption.py"])
    else:
        print("⏩ Caption Overlay is DISABLED in config.")
    
    yield
    
    # Clean up
    print("Shutting down...")
    if caption_process:
        caption_process.terminate()

# ----------------- FASTAPI SETUP -----------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- STATE MANAGEMENT -----------------
broadcasters: Dict[str, "ConnectionState"] = {}
viewers: Dict[str, WebSocket] = {}

class ConnectionState:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.buffer = np.zeros(0, dtype=np.float32)
        self.partial_history = []
        self.agreed_text = ""
        self.speech_active = False
        self.last_speech_ts = 0.0
        self.lock = asyncio.Lock()
        self.partial_task = None
        self.current_segment_start = None

    async def send_json(self, obj: Dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception as e:
            print(f"WebSocket send failed: {e}")

# ----------------- BROADCAST HELPER -----------------
async def broadcast_to_viewers(obj: Dict[str, Any]):
    if not viewers:
        return
    
    dead_connections = []
    payload = json.dumps(obj)
    
    for cid, ws in list(viewers.items()):
        try:
            await ws.send_text(payload)
        except Exception:
            dead_connections.append(cid)
            
    for cid in dead_connections:
        viewers.pop(cid, None)

# ----------------- TRANSCRIPTION LOGIC -----------------
def transcribe_audio(audio_np):
    global processor, model
    if audio_np is None or len(audio_np) < int(0.5 * config.SAMPLE_RATE):
        return ""
    try:
        inputs = processor(audio=audio_np, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(config.DEVICE)

        with torch.no_grad():
            ids = model.generate(
                input_features, 
                max_new_tokens=128, 
                language="vi", 
                task="transcribe",
                pad_token_id=processor.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.1,
                temperature=0.0,
                condition_on_prev_tokens=False,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""

async def do_transcribe_and_send(state: ConnectionState, final=False):
    async with state.lock:
        audio = state.buffer.copy()
    
    if audio.size == 0:
        return

    # Process audio
    if not final:
        audio_proc = process_realtime_chunk(audio)
    else:
        audio_proc = process_final_sentence(audio, sr=config.SAMPLE_RATE)

    # Run model in thread pool to avoid blocking async loop
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, lambda: transcribe_audio(audio_proc))
    except asyncio.CancelledError:
        return

    if not text:
        return

    # --- TRANSLATION ---
    try:
        trans_text = await loop.run_in_executor(None, lambda: translator.translate(text))
    except Exception:
        trans_text = ""

    if final:
        final_text = text
        # Send Final
        payload = {"type": "fullSentence", "text": final_text, "trans": trans_text}
        await state.send_json(payload)
        await broadcast_to_viewers(payload)
        
        # Reset buffer
        async with state.lock:
            state.buffer = np.zeros(0, dtype=np.float32)
            state.partial_history = []
            state.agreed_text = ""
    else:
        # Send Partial
        payload = {"type": "realtime", "text": text, "trans": trans_text}
        await state.send_json(payload)
        await broadcast_to_viewers(payload)

async def periodic_partial_sender(state: ConnectionState):
    try:
        while state.speech_active:
            await do_transcribe_and_send(state, final=False)
            await asyncio.sleep(config.PARTIAL_INTERVAL)
    except asyncio.CancelledError:
        return

# ----------------- WEBSOCKET ENDPOINT -----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = str(id(ws))
    role = ws.query_params.get("role", "broadcaster")

    # --- VIEWER ROLE ---
    if role == "viewer":
        viewers[cid] = ws
        print(f"Viewer connected ID={cid}")
        try:
            while True:
                # Keep connection alive
                try:
                    await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception:
                    await asyncio.sleep(0.1)
        finally:
            viewers.pop(cid, None)
            print(f"Viewer disconnected ID={cid}")
        return

    # --- BROADCASTER ROLE ---
    state = ConnectionState(ws)
    broadcasters[cid] = state
    print(f"Broadcaster connected ID={cid}")

    try:
        while True:
            # Receive audio data
            data = await ws.receive_bytes()
            
            # Parse custom protocol: [4 bytes len][json meta][pcm bytes]
            meta_len = struct.unpack_from("<I", data, 0)[0]
            audio_bytes = data[4 + meta_len:]
            
            if len(audio_bytes) == 0:
                continue
                
            # Convert bytes to float32
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            async with state.lock:
                state.buffer = np.concatenate([state.buffer, pcm])
                if len(state.buffer) > config.MAX_BUFFER_SAMPLES:
                    state.buffer = state.buffer[-config.MAX_BUFFER_SAMPLES:]
                    # print(f"Buffer trimmed: {len(state.buffer)} samples")

            # VAD Check
            prob = vad_prob_for_buffer(pcm)
            now = time.time()

            if prob >= config.VAD_THRESHOLD:
                if not state.speech_active:
                    state.speech_active = True
                    state.last_speech_ts = now
                    
                    # Notify Start
                    await state.send_json({"type": "vad_start"})
                    await broadcast_to_viewers({"type": "vad_start"})
                    print("Speech start")
                    
                    # Start partial transcription task
                    state.partial_task = asyncio.create_task(periodic_partial_sender(state))
                else:
                    state.last_speech_ts = now
            
            elif state.speech_active and (now - state.last_speech_ts) > config.SILENCE_LIMIT:
                # Silence detected -> Finalize
                state.speech_active = False
                if state.partial_task:
                    state.partial_task.cancel()
                    state.partial_task = None
                
                print("Speech end - finalizing")
                
                # Notify Stop
                await state.send_json({"type": "vad_stop"})
                await broadcast_to_viewers({"type": "vad_stop"})
                
                # Transcribe final sentence
                await do_transcribe_and_send(state, final=True)

    except WebSocketDisconnect:
        print(f"Broadcaster disconnected ID={cid}")
    finally:
        if state.partial_task:
            state.partial_task.cancel()
        broadcasters.pop(cid, None)

# ----------------- SERVE HTML -----------------
@app.get("/")
async def serve_viewer():
    with open("overlay_client.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ----------------- MAIN -----------------
if __name__ == "__main__":
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, log_level="info")
