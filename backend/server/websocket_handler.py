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

    Correct gate order:
        Energy gate (per chunk)
        → VAD gate (per chunk)
        → Segmenter
        → Speech-ratio gate (FINAL segment only)
        → Whisper
    """

    def __init__(self):
        self.vad = VADManager()
        self.asr = WhisperStreaming()
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

    # =========================================================
    # Init
    # =========================================================

    async def init(self):
        print("[Handler] Loading models...")
        self.vad.load()
        self.asr.load()
        print("[Handler] Ready!")

    # =========================================================
    # WebSocket router
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
    # Audio handling (REALTIME SAFE)
    # =========================================================

    async def _handle_audio(self, data):
        if not self.is_recording:
            return None

        audio_bytes = base64.b64decode(data.get("audio", ""))
        audio = (
            np.frombuffer(audio_bytes, np.int16)
            .astype(np.float32) / 32768.0
        )

        # ============================
        # 1️⃣ Energy gate
        # ============================

        rms = self.compute_rms(audio)
        if rms < 0.003:
            return None

        if not self.energy_is_consistent(audio):
            return None

        # ============================
        # 2️⃣ VAD gate (per chunk)
        # ============================

        is_speech = self.vad.is_speech(audio)

        # ============================
        # 3️⃣ Segmenter
        # ============================

        now_ts = datetime.now().timestamp()
        result = self.segmenter.process(audio, is_speech, now_ts)

        if result is None:
            return None

        kind, chunk = result
        sec = len(chunk) / SAMPLE_RATE
        print(f"\n🧩 Segment → {kind.upper()} | {sec:.2f}s")

        return self._decode_chunk(
            chunk=chunk,
            is_final=(kind == "final")
        )

    # =========================================================
    # Decode (SEGMENT LEVEL)
    # =========================================================

    def _decode_chunk(self, chunk: np.ndarray, is_final: bool):
        duration = len(chunk) / SAMPLE_RATE

        # ❗ PARTIAL cần đủ dài để tránh spam
        if not is_final and duration < MIN_DECODE_SEC:
            print("⚠️ Skip PARTIAL decode (too short)")
            return None

        # ⭐ FINAL: speech-ratio gate đặt ĐÚNG CHỖ
        if is_final:
            if not self.speech_ratio_gate(chunk):
                print("⚠️ Skip FINAL decode (low speech ratio)")
                return None

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

        # ============================
        # Local agreement
        # ============================

        stable, unstable = self.agreement.process(raw_vi)
        display = f"{stable} {unstable}".strip()

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
    # Utils
    # =========================================================

    def _get_timestamp(self):
        if not self.session_start:
            return "00:00"
        elapsed = (datetime.now() - self.session_start).total_seconds()
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    @staticmethod
    def compute_rms(audio: np.ndarray):
        return np.sqrt(np.mean(audio ** 2) + 1e-8)

    @staticmethod
    def energy_is_consistent(
        audio: np.ndarray,
        frame_ms: int = 20,
        min_cv: float = 0.18
    ) -> bool:
        frame_len = int(SAMPLE_RATE * frame_ms / 1000)
        if len(audio) < frame_len * 3:
            return True  # realtime chunk → đừng chặn

        frames = [
            audio[i:i + frame_len]
            for i in range(0, len(audio) - frame_len, frame_len)
        ]

        rms_vals = np.array([
            np.sqrt(np.mean(f ** 2) + 1e-8) for f in frames
        ])

        cv = rms_vals.std() / (rms_vals.mean() + 1e-8)
        return cv >= min_cv

    # =========================================================
    # ⭐ Speech ratio gate (SEGMENT ONLY)
    # =========================================================

    def speech_ratio_gate(
        self,
        audio: np.ndarray,
        frame_ms: int = 30,
        min_ratio: float = 0.25
    ) -> bool:
        frame_len = int(SAMPLE_RATE * frame_ms / 1000)
        if len(audio) < frame_len * 3:
            return True  # câu rất ngắn vẫn cho qua

        speech = 0
        total = 0

        for i in range(0, len(audio) - frame_len, frame_len):
            frame = audio[i:i + frame_len]
            if self.vad.is_speech(frame):
                speech += 1
            total += 1

        ratio = speech / max(total, 1)
        print(f"🔎 Speech ratio: {ratio:.2f}")
        return ratio >= min_ratio
