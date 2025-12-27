# import numpy as np
# from collections import deque
# from backend.config import SAMPLE_RATE

# class SlidingWindow:
#     """
#     Sliding window buffer for streaming ASR.
    
#     Logic:
#     1. Accumulate audio until we reach 'window_sec'.
#     2. After processing, 'slide' the window:
#        - Keep the last 'overlap_sec' (context).
#        - Discard the rest (processed).
#        - Wait for new audio to fill back up to 'window_sec'.
       
#     ⚠️ LATENCY NOTE:
#     The 'stride' (time between updates) = window_sec - overlap_sec.
#     For smoother streaming (lower latency), INCREASE overlap_sec.
#     E.g: window=5.0, overlap=4.0 => Updates every 1.0s.
#     """
    
#     def __init__(self, window_sec=5.0, overlap_sec=1.0):
#         self.window_sec = window_sec
#         self.overlap_sec = overlap_sec
        
#         self.window_samples = int(window_sec * SAMPLE_RATE)
#         self.overlap_samples = int(overlap_sec * SAMPLE_RATE)
        
#         # Buffer to accumulate audio chunks
#         self.buffer = deque()
#         self.total_samples = 0
    
#     def add_audio(self, audio_chunk: np.ndarray):
#         """Add new audio chunk to buffer"""
#         if audio_chunk.size == 0:
#             return
            
#         # Ensure flat float32 array (Critical for concatenation)
#         chunk = audio_chunk.flatten().astype(np.float32)
        
#         self.buffer.append(chunk)
#         self.total_samples += chunk.size
    
#     def has_window(self) -> bool:
#         """Check if we have enough audio for a full window"""
#         return self.total_samples >= self.window_samples
    
#     def get_window(self) -> np.ndarray:
#         """Get current window of audio (last window_sec seconds)"""
#         if self.total_samples == 0:
#             return np.array([], dtype=np.float32)
            
#         # Concatenate all buffered audio
#         # Note: Concatenation is relatively fast for buffers < 10s
#         full_audio = np.concatenate(list(self.buffer))
        
#         # If buffer is smaller than window, return what we have (padding handled by Whisper usually)
#         # But strictly for sliding window, we usually return exactly the window size
#         if self.total_samples < self.window_samples:
#             return full_audio
            
#         # Return exact window size from the end
#         return full_audio[-self.window_samples:]
    
#     def slide(self):
#         """
#         Slide the window forward.
#         Strategy: Keep overlap_samples, discard the head.
#         """
#         # Nếu chưa đủ dữ liệu để giữ lại thì không làm gì (hoặc clear hết)
#         if self.total_samples <= self.overlap_samples:
#             return
            
#         # 1. Merge current buffer
#         full_audio = np.concatenate(list(self.buffer))
        
#         # 2. Slice to keep only the tail (overlap)
#         overlap_audio = full_audio[-self.overlap_samples:]
        
#         # 3. Reset buffer with overlap chunk
#         self.buffer.clear()
#         self.buffer.append(overlap_audio)
#         self.total_samples = overlap_audio.size
    
#     def get_duration(self) -> float:
#         """Get current buffer duration in seconds"""
#         return self.total_samples / SAMPLE_RATE
    
#     def clear(self):
#         """Clear all buffered audio"""
#         self.buffer.clear()
#         self.total_samples = 0
        