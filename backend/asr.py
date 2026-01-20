import numpy as np
import logging
from backend.audio_normalizer import AdaptiveNormalizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class WhisperASR:
    """Faster-Whisper ASR using CTranslate2 backend"""
    
    def __init__(self, model_size="large-v3", language=None, device="cuda", cache_dir=None):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        
        self.normalizer = AdaptiveNormalizer(target_dB=-20.0, sample_rate=SAMPLE_RATE)
        logger.info("[ASR] Adaptive audio normalization enabled")

    def load_model(self):
        if self.model is not None:
            return
        
        from faster_whisper import WhisperModel
        
        logger.info(f"Loading Faster-Whisper: {self.model_size}")
        
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type,
            download_root=self.cache_dir,
        )
        logger.info("Faster-Whisper loaded")

    def transcribe(self, audio: np.ndarray, prompt: str = None) -> tuple:
        if self.model is None:
            self.load_model()
        
        normalized_audio = audio

        if len(audio) < 8000:
            return "", 0.0
        
        normalized_audio, norm_stats = self.normalizer.normalize(audio)

        if normalized_audio.dtype != np.float32:
            normalized_audio = normalized_audio.astype(np.float32)
        
        # Transcribe with normalized audio
        segments, info = self.model.transcribe(
            normalized_audio,
            language=self.language if self.language else None,
            task="transcribe",
            beam_size=5,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            without_timestamps=True,
        )
        
        # Collect segments
        text_parts = []
        total_prob = 0.0
        seg_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            total_prob += segment.avg_logprob
            seg_count += 1
        
        text = " ".join(text_parts).strip()
        
        # Calculate confidence
        conf = 0.0
        if seg_count > 0:
            avg_logprob = total_prob / seg_count
            conf = min(1.0, max(0.0, 1.0 + avg_logprob / 2))
        
        return text, conf


class StreamingASRProcessor:
    """Streaming processor for audio chunks"""
    
    def __init__(self, asr: WhisperASR, context_size: int = 3):
        self.asr = asr
        self.context_size = context_size
        self.audio_buffer = np.array([], dtype=np.float32)
        self.context_history = []

    def init(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.context_history = []

    def insert_audio(self, audio: np.ndarray):
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

    def process_partial(self, prompt: str = None) -> tuple:
        if len(self.audio_buffer) < SAMPLE_RATE:
            return "", 0.0
        
        # Whisper will normalize internally
        text, conf = self.asr.transcribe(self.audio_buffer, prompt)
        return text, conf

    def process_final(self, prompt: str = None) -> tuple:
        if len(self.audio_buffer) < 4000:
            self.audio_buffer = np.array([], dtype=np.float32)
            return "", 0.0
        
        # Whisper will normalize internally
        text, conf = self.asr.transcribe(self.audio_buffer, prompt)
        
        if text:
            self.context_history.append(text)
            if len(self.context_history) > self.context_size * 2:
                self.context_history = self.context_history[-self.context_size:]
        
        self.audio_buffer = np.array([], dtype=np.float32)
        return text, conf