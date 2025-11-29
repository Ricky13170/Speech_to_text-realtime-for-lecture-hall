import torch

# --- NETWORK ---
HOST = "127.0.0.1"
PORT = 8000
# Client connects to localhost by default, change IP if needed
WS_URL = f"ws://127.0.0.1:{PORT}/ws"

# --- AUDIO ---
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
MAX_BUFFER_SEC = 10
MAX_BUFFER_SAMPLES = MAX_BUFFER_SEC * SAMPLE_RATE

# --- MODEL ---
MODEL_ID = "vinai/PhoWhisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- VAD & LOGIC ---
VAD_THRESHOLD = 0.6
SILENCE_LIMIT = 0.8  # Tăng nhẹ lên 0.8s để tránh ngắt câu quá sớm
PARTIAL_INTERVAL = 0.1 # Giảm xuống 0.1s để chữ hiện ra siêu mượt (Real-time)

# --- UI ---
WINDOW_NAME = "ASR Live Caption"
# FONT_PATH = "C:/Windows/Fonts/arial.ttf" # Windows
FONT_PATH = "arial.ttf" # Cross-platform (if installed) or provide path

# --- SYSTEM ---
# Set False to disable caption.py (overlay window)
USE_CAPTION_OVERLAY = True 
