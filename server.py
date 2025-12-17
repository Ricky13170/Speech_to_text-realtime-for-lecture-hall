import asyncio
import json
import struct
import time
from typing import Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from vad_silero import vad_prob_for_buffer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ===== SIGNAL FRONTEND =====
from signal_condition.filters import highpass_filter, normalize_audio
from signal_condition.denoise import reduce_noise

# ================= MODEL =================
print(f"🔄 Loading model {config.MODEL_ID} on {config.DEVICE}")
processor = AutoProcessor.from_pretrained(config.MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(config.MODEL_ID).to(config.DEVICE)
model.eval()
print("✅ Model ready")

# ================= APP =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= VAD CONTROLLER =================
class VADController:
    def __init__(self):
        self.speech = False
        self.on_count = 0
        self.off_count = 0
        self.last_voice_ts = 0.0

    def update(self, prob: float, now: float):
        """
        return: "start" | "continue" | "stop" | None
        """

        # ----------- OFF STATE -----------
        if not self.speech:
            if prob >= config.VAD_ON:
                self.on_count += 1
            else:
                self.on_count = 0

            if self.on_count >= config.ON_DEBOUNCE_FRAMES:
                self.speech = True
                self.last_voice_ts = now
                self.on_count = 0
                self.off_count = 0
                return "start"

            return None

        # ----------- ON STATE -----------
        if prob >= config.VAD_OFF:
            self.last_voice_ts = now
            self.off_count = 0
            return "continue"

        self.off_count += 1

        if self.off_count >= config.OFF_DEBOUNCE_FRAMES:
            if (now - self.last_voice_ts) > config.MAX_SILENCE:
                self.speech = False
                self.off_count = 0
                return "stop"

        return None


# ================= STATE =================
class ConnectionState:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.buffer = np.zeros(0, np.float32)
        self.partial_history = []
        self.agreed_text = ""
        self.partial_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.vad = VADController()

    async def send(self, obj: Dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception:
            pass


# ================= AGREEMENT =================
def apply_local_agreement(state: ConnectionState, text: str):
    state.partial_history.append(text)
    if len(state.partial_history) > 6:
        state.partial_history.pop(0)

    votes = {}
    for t in state.partial_history:
        votes[t] = votes.get(t, 0) + 1

    best, count = max(votes.items(), key=lambda x: x[1])
    if count >= config.PARTIAL_MIN_VOTES and len(best) > len(state.agreed_text):
        state.agreed_text = best
        return best
    return None


# ================= SIGNAL FRONTEND =================
def frontend_realtime(audio: np.ndarray) -> np.ndarray:
    audio = highpass_filter(audio)
    audio = normalize_audio(audio)
    return audio


def frontend_final(audio: np.ndarray) -> np.ndarray:
    audio = frontend_realtime(audio)
    audio = reduce_noise(audio, sr=config.SAMPLE_RATE)
    return audio


# ================= TRANSCRIBE =================
def transcribe(audio: np.ndarray) -> str:
    if len(audio) < int(config.MIN_DECODE_SEC * config.SAMPLE_RATE):
        return ""

    inputs = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    feats = inputs.input_features.to(config.DEVICE)

    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=96)

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ================= PARTIAL LOOP =================
async def partial_loop(state: ConnectionState):
    try:
        while state.vad.speech:
            async with state.lock:
                audio = state.buffer.copy()

            audio = frontend_realtime(audio)

            txt = await asyncio.get_event_loop().run_in_executor(
                None, lambda: transcribe(audio)
            )

            if txt:
                committed = apply_local_agreement(state, txt)
                await state.send({
                    "type": "realtime",
                    "text": committed or txt
                })

            await asyncio.sleep(config.PARTIAL_INTERVAL)

    except asyncio.CancelledError:
        pass


# ================= WEBSOCKET =================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state = ConnectionState(ws)

    try:
        while True:
            data = await ws.receive_bytes()
            meta_len = struct.unpack_from("<I", data, 0)[0]
            pcm_bytes = data[4 + meta_len:]

            pcm = (
                np.frombuffer(pcm_bytes, np.int16)
                .astype(np.float32) / 32768.0
            )

            async with state.lock:
                state.buffer = np.concatenate([state.buffer, pcm])
                if len(state.buffer) > config.MAX_BUFFER_SAMPLES:
                    state.buffer = state.buffer[-config.MAX_BUFFER_SAMPLES:]

            prob = vad_prob_for_buffer(pcm)
            now = time.time()
            event = state.vad.update(prob, now)

            # ===== SPEECH START =====
            if event == "start":
                state.partial_task = asyncio.create_task(partial_loop(state))
                await state.send({"type": "vad_start"})

            # ===== SPEECH END =====
            elif event == "stop":
                if state.partial_task:
                    state.partial_task.cancel()
                    state.partial_task = None

                async with state.lock:
                    audio = state.buffer.copy()
                    state.buffer = np.zeros(0, np.float32)
                    state.partial_history = []
                    state.agreed_text = ""

                audio = frontend_final(audio)

                txt = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: transcribe(audio)
                )

                if txt:
                    await state.send({"type": "fullSentence", "text": txt})

                await state.send({"type": "vad_stop"})

    except WebSocketDisconnect:
        pass
    finally:
        if state.partial_task:
            state.partial_task.cancel()


# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
