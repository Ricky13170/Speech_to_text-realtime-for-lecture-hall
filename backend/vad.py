import numpy as np
from typing import Tuple


class EnergyVAD:
    """Simple energy-based Voice Activity Detection"""
    
    def __init__(self, threshold: float = 0.01, min_speech_duration: float = 0.1):
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.sample_rate = 16000
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate Root Mean Square energy"""
        if len(audio) == 0:
            return 0.0
        return np.sqrt(np.mean(audio ** 2))
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech based on energy"""
        rms = self.calculate_rms(audio)
        return rms > self.threshold
    
    def get_speech_segments(self, audio: np.ndarray) -> list:
        """Get list of (start, end) tuples for speech segments"""
        chunk_size = int(self.sample_rate * 0.02)  # 20ms chunks
        segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            is_speech = self.is_speech(chunk)
            
            if is_speech and not in_speech:
                speech_start = i
                in_speech = True
            elif not is_speech and in_speech:
                if (i - speech_start) / self.sample_rate >= self.min_speech_duration:
                    segments.append((speech_start, i))
                in_speech = False
        
        if in_speech:
            if (len(audio) - speech_start) / self.sample_rate >= self.min_speech_duration:
                segments.append((speech_start, len(audio)))
        
        return segments
