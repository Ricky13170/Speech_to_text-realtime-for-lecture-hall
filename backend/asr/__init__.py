# E2E ASR - Whisper handles both transcription and translation
from backend.asr.local_agreement import LocalAgreement
from backend.asr.whisper_streaming import WhisperStreaming

__all__ = ["LocalAgreement", "WhisperStreaming"]
