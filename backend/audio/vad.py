import numpy as np
from backend.config import SAMPLE_RATE

class VADManager:
    """
    Adaptive Energy-based VAD (Voice Activity Detection).
    Tự động điều chỉnh ngưỡng dựa trên độ ồn môi trường.
    """
    
    def __init__(self):
        # Cấu hình ngưỡng cơ bản
        self.base_threshold = 0.015
        
        # Biến để theo dõi độ ồn môi trường (Noise Floor)
        self.noise_level = 0.005  
        self.alpha = 0.05  
        
        self._loaded = True
    
    def load(self):
        pass  # Không cần load model nặng
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Kiểm tra xem chunk âm thanh có tiếng người không.
        Sử dụng thuật toán thích ứng (Adaptive Thresholding).
        """
        if len(audio) == 0:
            return False
            
        # 1. Tính năng lượng RMS (Root Mean Square)
        # Sử dụng float64 cho tính toán trung gian để tránh tràn số
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        
        # 2. Tính ngưỡng động (Dynamic Threshold)
        # Ngưỡng kích hoạt = Độ ồn nền + Biên độ an toàn (base_threshold)
        current_threshold = self.noise_level + self.base_threshold
        
        # 3. Cập nhật độ ồn nền (Noise Floor Update)
        # Nếu âm thanh nhỏ (đang im lặng hoặc chỉ có tiếng quạt), cập nhật lại mức noise
        if rms < current_threshold:
            # Chạy trung bình động (Moving Average) để bám sát môi trường
            self.noise_level = (1 - self.alpha) * self.noise_level + self.alpha * rms
        
        # Giới hạn noise level không được quá cao (tránh trường hợp ồn quá threshold tăng vọt)
        self.noise_level = min(self.noise_level, 0.05)
        
        # 4. Quyết định
        # Là tiếng nói khi RMS vượt qua ngưỡng động
        is_speech = rms > current_threshold
        
        # Debug log (nếu cần test ngưỡng)
        # print(f"RMS: {rms:.4f} | Noise: {self.noise_level:.4f} | Thr: {current_threshold:.4f} | Speech: {is_speech}")
        
        return is_speech

    def reset(self):
        """Reset lại trạng thái khi bắt đầu session mới"""
        self.noise_level = 0.005
        
