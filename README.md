# ASR Realtime - Nhận dạng giọng nói thời gian thực

Hệ thống chuyển giọng nói thành văn bản theo thời gian thực, hỗ trợ dịch Việt - Anh.

## Tính năng

- Nhận dạng giọng nói tiếng Việt realtime
- Dịch tự động sang tiếng Anh
- Giao diện web xem transcript
- Hỗ trợ deploy lên Modal (GPU cloud)

## Cài đặt

```bash
# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Sử dụng

### Cách 1: Dùng Modal Cloud (Khuyến nghị)

**Bước 1:** Đăng nhập Modal

```bash
pip install modal
modal token new
```

**Bước 2:** Deploy server

```bash
modal deploy modal_app.py
```

**Bước 3:** Chạy client

```bash
python client.py
```

Client sẽ hiện link để mở trình duyệt xem transcript.

### Cách 2: Chạy local (Cần GPU)

**Bước 1:** Chạy server

```bash
python server.py
```

**Bước 2:** Chạy client

```bash
python client.py
```

**Bước 3:** Mở trình duyệt

```text
http://localhost:8000
```

## Cấu hình

Sửa file `config.py`:

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `USE_MODAL` | Dùng Modal cloud | `True` |
| `MODEL_ID` | Model ASR | `openai/whisper-large-v3` |
| `VAD_THRESHOLD` | Độ nhạy phát hiện giọng nói | `0.6` |
| `SILENCE_LIMIT` | Thời gian im lặng để kết thúc câu | `0.8s` |

## Cấu trúc project

```text
├── client.py          # Thu âm và gửi audio
├── server.py          # Server local (cần GPU)
├── modal_app.py       # Server trên Modal cloud
├── audio_processor.py # Xử lý audio, VAD
├── config.py          # Cấu hình
├── overlay_client.html # Giao diện web viewer
└── requirements.txt   # Dependencies
```

## Yêu cầu

- Python 3.10+
- Microphone
- GPU NVIDIA (nếu chạy local)
- Tài khoản Modal (nếu dùng cloud)
