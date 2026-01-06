import numpy as np
import time
import torch
import hashlib
from collections import deque

from backend.config import SAMPLE_RATE, WHISPER_DEVICE

WHISPER_E2E_MODEL = "openai/whisper-large-v3"


# ======================= Hallucination Filter =======================

class HallucinationFilter:
    def __init__(self, history_size=5):
        self.recent_hashes = deque(maxlen=history_size)
        self.recent_texts = deque(maxlen=history_size)

    def _hash(self, text):
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]

    def is_hallucination(self, text: str, amp: float):
        if not text or len(text.split()) < 2:
            return True, "empty"

        words = text.split()

        if amp < 0.02 and len(words) > 4:
            return True, "quiet_audio_long_text"

        h = self._hash(text)
        if h in self.recent_hashes:
            return True, "repeat"

        self.recent_hashes.append(h)
        self.recent_texts.append(text)
        return False, "ok"

    def reset(self):
        self.recent_hashes.clear()
        self.recent_texts.clear()


# ======================= Whisper Segment ASR =======================

class WhisperStreaming:
    """
    Segment-based Whisper ASR (VAD-driven)
    Stateless audio, only text context
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = WHISPER_DEVICE
        self.segment_id = 0
        self.context_prompt = None
        self.filter = HallucinationFilter()
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"[ASR] Loading {WHISPER_E2E_MODEL}...")
        start = time.time()

        self.processor = WhisperProcessor.from_pretrained(WHISPER_E2E_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_E2E_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()
        self._loaded = True
        print(f"[ASR] Loaded in {time.time() - start:.1f}s")

    def reset(self):
        self.segment_id = 0
        self.context_prompt = None
        self.filter.reset()

    def set_context(self, text: str):
        if text:
            self.context_prompt = text[-200:]

    def transcribe(self, audio: np.ndarray, timestamp: str, final: bool):
        max_amp = float(np.max(np.abs(audio)))
        if max_amp < 0.002:
            return {}

        start = time.time()

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(self.device).to(torch.float16)

        with torch.no_grad():
            vi_ids = self.model.generate(
                input_features,
                language="vi",
                task="transcribe",
                max_length=128,
                do_sample=False,
                num_beams=1,
            )

        vi_text = self.processor.batch_decode(
            vi_ids, skip_special_tokens=True
        )[0].strip()

        is_hallu, reason = self.filter.is_hallucination(vi_text, max_amp)
        if is_hallu:
            print(f"⚠️ Hallucination filtered ({reason})")
            return {}

        with torch.no_grad():
            en_ids = self.model.generate(
                input_features,
                language="vi",
                task="translate",
                max_length=128,
                do_sample=False,
                num_beams=1,
            )

        en_text = self.processor.batch_decode(
            en_ids, skip_special_tokens=True
        )[0].strip()

        self.segment_id += 1

        return {
            "segment_id": self.segment_id,
            "source": vi_text,
            "target": en_text,
            "timestamp": timestamp,
            "processing_ms": int((time.time() - start) * 1000)
        }
