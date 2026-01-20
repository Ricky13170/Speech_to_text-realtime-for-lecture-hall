# 🎤 Real-time Vietnamese-English Speech Translation

Hệ thống dịch tiếng nói Việt-Anh thời gian thực, sử dụng cho giảng đường.

## 🚀 Quick Start

```bash
# 1. Clone và cài đặt
git clone https://github.com/Senju14/Speech_to_text-realtime-for-lecture-hall.git
cd Speech_to_text-realtime-for-lecture-hall

# 2. Setup Modal CLI
pip install modal
modal token new

# 3. Deploy
modal deploy main.py
```

Truy cập URL được in ra sau khi deploy.

## 📐 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                       │
├─────────────────────────────────────────────────────────────────┤
│  🎤 Microphone ──► AudioWorklet ──► Resample 16kHz ──► Base64   │
│                                                          │       │
│  📊 UI Manager ◄── WebSocket ◄───────────────────────────┘       │
│     └── Transcript Display (Vi + En)                             │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Modal GPU)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Audio ──► VAD (Energy) ──► Buffer ──► Faster-Whisper ──► Text  │
│                                              │                    │
│                              ┌───────────────┘                    │
│                              ▼                                    │
│                    Hallucination Filter                           │
│                    (Pattern + WPS + Confidence)                   │
│                              │                                    │
│                              ▼                                    │
│                    NLLB Translator ──► English Text               │
│                              │                                    │
│                              ▼                                    │
│                    WebSocket Response                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| ASR | Faster-Whisper (large-v3) | Vietnamese speech recognition |
| Translation | NLLB 3.3B | Vi→En neural machine translation |
| VAD | Energy-based RMS | Voice activity detection |
| Streaming | WebSocket | Real-time bidirectional |
| Backend | Modal + FastAPI | Serverless GPU compute |
| Frontend | Vanilla JS + CSS | Lightweight UI |

## 📁 Cấu trúc dự án

```
├── main.py                 # Modal entry point
├── backend/
│   ├── asr.py             # Faster-Whisper ASR
│   ├── translation.py     # NLLB translator
│   ├── handler.py         # WebSocket session handler
│   ├── config.py          # Configuration
│   └── vad.py             # Voice Activity Detection
└── frontend/
    ├── index.html         # Main UI
    ├── style.css          # Styling
    └── js/
        ├── main.js        # App controller
        ├── audio.js       # Audio capture
        ├── socket.js      # WebSocket client
        └── ui.js          # UI manager
```

## ⚙️ Cấu hình

Chỉnh sửa `backend/config.py`:

```python
WHISPER_MODEL = "large-v3"      # Model size
WHISPER_LANGUAGE = "vi"         # Force Vietnamese
MODAL_GPU = "A100"              # GPU type
VAD_THRESHOLD = 0.01            # Voice detection sensitivity
MAX_BUFFER_DURATION = 8.0       # Max audio buffer (seconds)
```

## 🔬 Phương pháp chính (cho Paper)

1. **Streaming ASR Pipeline**
   - Chunk-based processing với VAD
   - Faster-Whisper cho low-latency

2. **Hallucination Detection**
   - Pattern matching (YouTube artifacts)
   - Words-per-second validation
   - Confidence thresholding

3. **Cascade Translation**
   - NLLB 3.3B với safetensors
   - Async translation pipeline

4. **Real-time WebSocket Protocol**
   - Binary audio streaming
   - JSON transcript responses

## 📊 Metrics

- **Latency**: ~0.5-1s (partial), ~2-3s (final + translation)
- **GPU**: A100 40GB
- **Model load**: ~25s cold start

## 📈 Evaluation Results

Streaming ASR benchmark on 100 samples per dataset:

| Model | Dataset | GPU | WER | CER | TTFT | RTF |
|-------|---------|-----|-----|-----|------|-----|
| Whisper | vlsp2020 | A100 | 26.94% | 22.19% | 4ms | 0.070x |
| Whisper | earnings22 | A100 | 25.44% | 19.65% | 1ms | 0.060x |
| PhoWhisper | vlsp2020 | A100 | **16.16%** | **14.82%** | 4ms | 0.081x |
| PhoWhisper | earnings22 | A100 | 29.59% | 21.80% | 2ms | 0.088x |

**Key findings:**
- PhoWhisper achieves **16.16% WER** on Vietnamese (VLSP2020) - 40% better than Whisper
- Whisper performs better on English (earnings22)
- RTF ~0.06-0.09x = **~11-17x faster than real-time**

See `test/README.md` for running evaluations.

## 📝 License

MIT License

