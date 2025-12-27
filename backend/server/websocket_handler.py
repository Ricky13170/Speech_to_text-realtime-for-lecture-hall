import json
import base64
import numpy as np
from datetime import datetime

from backend.audio import VADManager
from backend.audio.speech_segment_buffer import SpeechSegmentBuffer
from backend.asr import WhisperStreaming, LocalAgreement
from backend.config import (
    SAMPLE_RATE,
    SILENCE_LIMIT,
    MAX_SEGMENT_SEC,
    OVERLAP_SEC,
    MIN_DECODE_SEC,
)


class WebSocketHandler:
    """
    WebSocket Handler
    Architecture: Segment-based (VAD-driven)
    """

    def __init__(self):
        self.vad = VADManager()
        self.asr = WhisperStreaming()          # Whisper E2E (segment-based)
        self.agreement = LocalAgreement()

        self.segmenter = SpeechSegmentBuffer(
            sample_rate=SAMPLE_RATE,
            max_sec=MAX_SEGMENT_SEC,
            overlap_sec=OVERLAP_SEC,
            silence_limit=SILENCE_LIMIT,
        )

        self.is_recording = False
        self.session_start = None
        self.last_stable = ""

    async def init(self):
        print("[Handler] Loading models...")
        self.vad.load()
        self.asr.load()
        print("[Handler] Ready!")

    # =========================================================
    # WebSocket message router
    # =========================================================

    async def handle(self, message: str):
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "start":
                return self._handle_start()

            if msg_type == "stop":
                return self._handle_stop()

            if msg_type == "audio":
                return await self._handle_audio(data)

            if msg_type == "context":
                return self._handle_context(data)

            if msg_type == "ping":
                return json.dumps({"type": "pong"})

        except Exception as e:
            print(f"[Handler] Error: {e}")
            return json.dumps({"type": "error", "message": str(e)})

    # =========================================================
    # Session control
    # =========================================================

    def _handle_start(self):
        print("[Handler] START SESSION")

        self.is_recording = True
        self.session_start = datetime.now()

        self.segmenter.reset()
        self.vad.reset()
        self.asr.reset()
        self.agreement.reset()
        self.last_stable = ""

        return json.dumps({"type": "status", "status": "recording"})

    def _handle_stop(self):
        print("[Handler] STOP SESSION")

        self.is_recording = False
        duration = (
            (datetime.now() - self.session_start).total_seconds()
            if self.session_start else 0
        )

        return json.dumps({
            "type": "transcript",
            "segment_id": self.asr.segment_id,
            "source": "",
            "target": "",
            "timestamp": self._get_timestamp(),
            "is_final": True,
            "session_duration": duration
        })

    def _handle_context(self, data):
        return json.dumps({"type": "status", "status": "context_set"})

    # =========================================================
    # Audio handling (CORE)
    # =========================================================

    async def _handle_audio(self, data):
        if not self.is_recording:
            return None

        audio_bytes = base64.b64decode(data.get("audio", ""))
        audio = (
            np.frombuffer(audio_bytes, np.int16)
            .astype(np.float32) / 32768.0
        )

        now_ts = datetime.now().timestamp()
        is_speech = self.vad.is_speech(audio)

        result = self.segmenter.process(audio, is_speech, now_ts)
        if result is None:
            return None

        kind, chunk = result   # kind = "partial" | "final"

        sec = len(chunk) / SAMPLE_RATE
        print(f"\n🧩 Segment → {kind.upper()} | {sec:.2f}s")

        return self._decode_chunk(
            chunk=chunk,
            is_final=(kind == "final")
        )

    # =========================================================
    # Decode
    # =========================================================

    def _decode_chunk(self, chunk: np.ndarray, is_final: bool):
        duration = len(chunk) / SAMPLE_RATE
        if duration < MIN_DECODE_SEC:
            print("⚠️ Skip decode (too short)")
            return None

        # 🚫 KHÔNG reset_audio
        # 🚫 KHÔNG add_audio
        # ✅ Whisper nhận audio nguyên khối

        result = self.asr.transcribe(
            audio=chunk,
            timestamp=self._get_timestamp(),
            final=is_final
        )

        if not result or not result.get("source"):
            return None

        raw_vi = result["source"].strip()
        raw_en = result.get("target", "").strip()

        print(f"\n📝 {'FINAL' if is_final else 'PARTIAL'} [VI] {raw_vi}")
        if raw_en:
            print(f"      [EN] {raw_en}")

        # ===== Local Agreement =====
        stable, unstable = self.agreement.process(raw_vi)
        display = f"{stable} {unstable}".strip()

        # ===== Context update (FINAL only) =====
        if is_final and stable:
            self.asr.set_context(stable)
            self.last_stable = stable
            self.agreement.reset()

        return json.dumps({
            "type": "transcript",
            "segment_id": result.get("segment_id", 0),
            "source": display,
            "target": raw_en,
            "timestamp": result.get("timestamp"),
            "is_final": is_final,
            "processing_ms": result.get("processing_ms", 0)
        })

    # =========================================================

    def _get_timestamp(self):
        if not self.session_start:
            return "00:00"
        elapsed = (datetime.now() - self.session_start).total_seconds()
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
