import numpy as np
from scipy import signal
import logging

from backend.config import (
    SAMPLE_RATE, NOISE_REDUCE_ENABLED, NOISE_REDUCE_PROP_DECREASE,
    HIGHPASS_ENABLED, HIGHPASS_CUTOFF_HZ, NORMALIZE_ENABLED
)

class AudioPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 1. Setup High-pass Filter (Lọc tần số thấp gây ù)
        if HIGHPASS_ENABLED:
            try:
                nyquist = SAMPLE_RATE / 2
                cutoff = HIGHPASS_CUTOFF_HZ / nyquist
                # Dùng order thấp (2 hoặc 4) để tính toán nhanh hơn
                self.b, self.a = signal.butter(4, cutoff, btype='high')
            except Exception as e:
                self.logger.error(f"Failed to init highpass filter: {e}")
                self.b, self.a = None, None
        else:
            self.b, self.a = None, None

        # Cảnh báo nếu bật Noise Reduce
        if NOISE_REDUCE_ENABLED:
            self.logger.warning("WARNING: Server-side Noise Reduction is ENABLED. This causes high latency!")

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
            
        audio = audio.astype(np.float32)

        # 2. Apply High-pass Filter
        # Chỉ lọc nếu chunk đủ dài (để tránh lỗi scipy)
        if self.b is not None and len(audio) > 18: 
            try:
                audio = signal.filtfilt(self.b, self.a, audio).astype(np.float32)
            except Exception:
                pass

        # 3. Noise Reduction (KHUYÊN DÙNG: OFF)
        if NOISE_REDUCE_ENABLED:
            try:
                import noisereduce as nr
                # Giảm n_fft để chạy nhanh hơn trên chunk nhỏ
                audio = nr.reduce_noise(
                    y=audio, sr=SAMPLE_RATE,
                    prop_decrease=NOISE_REDUCE_PROP_DECREASE,
                    n_fft=512, # Tối ưu tốc độ
                    stationary=True 
                ).astype(np.float32)
            except Exception as e:
                print(f"[Preprocess] NR Error: {e}")

        # 4. Normalize (Optimized: Soft Clipping)
        # đảm bảo âm thanh không bị vỡ (clipping)
        if NORMALIZE_ENABLED:
            max_val = np.max(np.abs(audio))
            
            # Nếu tín hiệu quá lớn (gần vỡ tiếng), giảm gain xuống
            if max_val > 1.0:
                audio = audio / max_val * 0.95
            
        return audio
        
