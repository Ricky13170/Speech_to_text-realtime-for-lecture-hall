# Real-time Speech-to-Text for Lecture Halls

Hệ thống nhận dạng tiếng nói thời gian thực (Client-Server), sử dụng PhoWhisper và Silero VAD. Hỗ trợ phụ đề song ngữ Việt - Anh.

## 1. Cài đặt môi trường

Khuyên dùng `uv` để cài đặt nhanh hơn.

```bash
pip install uv
uv venv
.venv\Scripts\activate
```

## 2. Cài đặt thư viện

**Quan trọng:** Cài đặt PyTorch bản hỗ trợ GPU trước, sau đó mới cài các thư viện khác.

**Bước 1: Cài đặt PyTorch (CUDA 11.8)**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Bước 2: Cài đặt các thư viện còn lại**
```bash
uv pip install -r requirements.txt
```

## 3. Hướng dẫn chạy

**Bước 1: Khởi động Server** (Xử lý AI & Dịch thuật)
```bash
python server.py
```
*   Truy cập Web xem phụ đề: `http://localhost:8000`

**Bước 2: Khởi động Client** (Thu âm từ Microphone)
Mở một terminal khác:
```bash
python client.py
```

## 4. Cấu hình (`config.py`)

*   `USE_CAPTION_OVERLAY`: Đặt `True` để bật cửa sổ phụ đề nổi trên Desktop.
*   `MODEL_ID`: Đổi model AI (VD: `vinai/PhoWhisper-small`).
*   `VAD_THRESHOLD`: Độ nhạy bắt giọng nói (Mặc định 0.6).

> **Lưu ý:** Hệ thống tự động ghi log file `.vtt` vào thư mục gốc để phục vụ đánh giá độ chính xác.
