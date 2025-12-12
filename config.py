import torch

# === SERVER ===
HOST = "0.0.0.0"
PORT = 8000

# === MODAL CLOUD ===
USE_MODAL = True
MODAL_URL = "wss://huynguyenwork14--asr-whisper-large-v3-web-app.modal.run"
WS_URL = f"{MODAL_URL}/ws" if USE_MODAL else f"ws://127.0.0.1:{PORT}/ws"

# === AUDIO ===
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
MAX_BUFFER_SEC = 10
MAX_BUFFER_SAMPLES = MAX_BUFFER_SEC * SAMPLE_RATE

# === MODEL ===
MODEL_ID = "vinai/PhoWhisper-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === VAD ===
VAD_THRESHOLD = 0.6
SILENCE_LIMIT = 0.4
PARTIAL_INTERVAL = 0.1

# === UI ===
WINDOW_NAME = "ASR Live Caption"
FONT_PATH = "arial.ttf"
USE_CAPTION_OVERLAY = False
