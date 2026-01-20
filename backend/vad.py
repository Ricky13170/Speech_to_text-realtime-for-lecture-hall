import numpy as np
from backend.config import SAMPLE_RATE

class VADManager:
    def __init__(self, base_threshold: float = 0.015, alpha: float = 0.05):
        self.base_threshold = base_threshold
        self.alpha = alpha
        
        self.noise_level = 0.005  
        
        # Sanity limits
        self.min_noise = 0.001  
        self.max_noise = 0.05  
        
        self._loaded = True
    
    def load(self):
        """Compatibility method (no model to load)"""
        pass
    
    def is_speech(self, audio: np.ndarray) -> bool:
        if len(audio) == 0:
            return False
        
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        
        current_threshold = self.noise_level + self.base_threshold

        if rms < current_threshold:

            self.noise_level = (1 - self.alpha) * self.noise_level + self.alpha * rms

            self.noise_level = np.clip(self.noise_level, self.min_noise, self.max_noise)

        is_speech = rms > current_threshold
        
        return is_speech
    
    def reset(self):
        """Reset noise floor estimate (call at session start)"""
        self.noise_level = 0.005
    
    def get_stats(self) -> dict:
        return {
            "noise_level": self.noise_level,
            "noise_dB": 20 * np.log10(self.noise_level + 1e-10),
            "threshold": self.noise_level + self.base_threshold,
            "threshold_dB": 20 * np.log10(self.noise_level + self.base_threshold + 1e-10)
        }