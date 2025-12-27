import numpy as np
from backend.config import SAMPLE_RATE

class AudioBuffer:
    """
    Optimized AudioBuffer using List accumulation (O(1) append)
    instead of repeated NumPy concatenation (O(N^2) copy).
    """
    def __init__(self):
        # Tối ưu: Dùng list để chứa các chunk, append cực nhanh
        self.chunks = [] 
        self.start_time = 0.0
        self.total_samples = 0
    
    @property
    def duration(self) -> float:
        return self.total_samples / SAMPLE_RATE
    
    @property
    def timestamp(self) -> str:
        """Trả về thời gian BẮT ĐẦU của đoạn buffer hiện tại (định dạng MM:SS)"""
        # Logic cũ của bạn là start + duration (thời gian kết thúc). 
        # Chuẩn subtitle thường là thời gian bắt đầu.
        # Nếu bạn muốn thời gian kết thúc thì dùng: total_sec = int(self.start_time + self.duration)
        
        total_sec = int(self.start_time)
        minutes = total_sec // 60
        seconds = total_sec % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def append(self, audio: np.ndarray):
        """Thêm audio vào cuối buffer (O(1) complexity)"""
        if audio.size == 0:
            return
            
        # Chỉ append vào list, không concatenate ngay
        flat_audio = audio.flatten()
        self.chunks.append(flat_audio)
        self.total_samples += flat_audio.size
    
    def get_audio(self) -> np.ndarray:
        """Lấy toàn bộ audio ra thành 1 mảng numpy"""
        if not self.chunks:
            return np.array([], dtype=np.float32)
        
        # Chỉ nối khi thực sự cần dùng
        # Cache lại nếu cần (ở đây đơn giản hóa là nối luôn)
        return np.concatenate(self.chunks)
    
    def trim(self, keep_sec: float):
        """Giữ lại keep_sec giây cuối cùng"""
        keep_samples = int(keep_sec * SAMPLE_RATE)
        
        if self.total_samples <= keep_samples:
            return

        # Khi cần trim, bắt buộc phải merge lại để cắt chính xác
        full_audio = self.get_audio()
        
        # Tính thời gian đã bị cắt đi để cộng dồn vào start_time
        removed_samples = self.total_samples - keep_samples
        self.start_time += removed_samples / SAMPLE_RATE
        
        # Cắt lấy phần đuôi
        new_audio = full_audio[-keep_samples:]
        
        # Reset chunks chỉ còn 1 chunk mới đã cắt
        self.chunks = [new_audio]
        self.total_samples = new_audio.size
    
    def clear(self):
        """Xóa sạch buffer"""
        self.chunks = []
        self.total_samples = 0
        self.start_time = 0.0
        