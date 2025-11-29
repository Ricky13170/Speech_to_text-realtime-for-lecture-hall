# Real-time Speech-to-Text for Lecture Halls

Hệ thống nhận dạng tiếng nói thời gian thực mô hình Client-Server, tối ưu cho giảng đường sử dụng PhoWhisper và Silero VAD. Hỗ trợ hiển thị phụ đề song ngữ (Việt - Anh) qua giao diện Overlay hoặc Webcam.

## 1. Cài đặt (Khuyên dùng uv)

```bash
pip install uv
uv venv
.venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

## 2. Kiểm tra & Cấu hình GPU (Laptop)

Để hệ thống chạy mượt (Real-time), bắt buộc nên dùng GPU NVIDIA.

**Bước 1: Kiểm tra máy đã nhận GPU chưa**
Chạy script kiểm tra có sẵn:
```bash
python utils/check_gpu.py
```
*   Nếu hiện `CUDA Available: True` và tên Card rời (VD: RTX 3050) -> **Sẵn sàng**.
*   Nếu hiện `False` hoặc tên Card Onboard -> Cần cài lại PyTorch CUDA.

**Bước 2: Cài đặt PyTorch hỗ trợ CUDA (Nếu chưa nhận)**
Chạy lệnh sau để cài bản hỗ trợ GPU (Ví dụ cho CUDA 11.8/12.x):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. Hướng dẫn chạy

Hệ thống gồm 2 thành phần chính: **Server** (xử lý AI) và **Client** (thu âm).

**Bước 1: Khởi động Server**
```bash
python server.py
```
*   Server sẽ tải model AI (lần đầu chạy sẽ hơi lâu).
*   Sau khi load xong, giao diện **Caption Overlay** sẽ tự động hiện lên.
*   **Giao diện Caption:**
    *   Mặc định là thanh phụ đề trong suốt (Overlay).
    *   Bấm nút **📷 Bật Camera** để chuyển sang chế độ Webcam + Phụ đề.
    *   Bấm nút **⚙ Cài đặt** để chỉnh cỡ chữ, độ mờ nền.
    *   Kéo thả góc dưới phải để thay đổi kích thước cửa sổ.

**Bước 2: Khởi động Client (Microphone)**
Mở một terminal khác và chạy:
```bash
python client.py
```
*   Client sẽ thu âm từ microphone và gửi về Server.
*   Phụ đề sẽ hiện ra ngay lập tức trên cửa sổ Caption.

*Ngoài ra, sinh viên có thể xem phụ đề qua Web tại: `http://localhost:8000`*

## 4. Cấu hình
Sửa file `config.py` để thay đổi các thông số:
*   `USE_CAPTION_OVERLAY`: Đặt `True` để tự động bật giao diện Caption, `False` để tắt.
*   `MODEL_ID`: Đổi model AI (VD: `vinai/PhoWhisper-small`).
*   `VAD_THRESHOLD`: Độ nhạy bắt giọng nói (0.6 là mức khuyến nghị).
*   `PARTIAL_INTERVAL`: Tốc độ cập nhật phụ đề (0.1s cho độ trễ thấp nhất).
