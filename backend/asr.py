import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class WhisperASR:
    def __init__(self, model_size="large-v3", language=None, device="cuda", cache_dir=None):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.cache_dir = cache_dir
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        
        import whisper
        
        logger.info(f"Loading Whisper: {self.model_size}")
        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
            download_root=self.cache_dir
        )
        logger.info("Whisper loaded")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, prompt: str = None) -> tuple:
        if self.model is None:
            self.load_model()
        
        if len(audio) < 8000:
            return "", 0.0
        
        result = self.model.transcribe(
            audio,
            language=self.language if self.language else None,
            task="transcribe",
            fp16=True,
            temperature=0.0,
            best_of=1,
            beam_size=5,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            without_timestamps=True,
        )
        
        text = result.get("text", "").strip()
        
        segments = result.get("segments", [])
        conf = 0.0
        if segments:
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            
            if no_speech > 0.6 or avg_logprob < -1.0:
                return "", 0.0
            
            conf = min(1.0, max(0.0, 1.0 + avg_logprob))
        
        return text, conf


class StreamingASRProcessor:
    def __init__(self, asr: WhisperASR, context_size: int = 2):
        self.asr = asr
        self.audio_buffer = np.array([], dtype=np.float32)
        self.confirmed_text = []
        self.context_size = context_size

    def init(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.confirmed_text = []

    def get_context(self) -> str:
        if not self.confirmed_text:
            return None
        return " ".join(self.confirmed_text[-self.context_size:])

    def insert_audio(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if len(self.audio_buffer) > 480000:
            self.audio_buffer = self.audio_buffer[-480000:]

    def process_partial(self) -> tuple:
        if len(self.audio_buffer) < 8000:
            return "", 0.0
        return self.asr.transcribe(self.audio_buffer, prompt=self.get_context())

    def process_final(self) -> tuple:
        if len(self.audio_buffer) < 4000:
            return "", 0.0
            
        text, conf = self.asr.transcribe(self.audio_buffer, prompt=self.get_context())
        
        if text:
            self.confirmed_text.append(text)
            
        self.audio_buffer = np.array([], dtype=np.float32)
        return text, conf
