import sys
import json
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QHBoxLayout, QFrame, 
                             QGraphicsDropShadowEffect, QSizeGrip, QDialog, QSlider, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QUrl, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap
from PyQt5.QtWebSockets import QWebSocket

import config

# --- THREAD XỬ LÝ CAMERA (Để không lag giao diện) ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # Mở Camera mặc định (index 0)
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # Convert từ BGR (OpenCV) sang RGB (Qt)
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- HỘP THOẠI CÀI ĐẶT ---
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt hiển thị")
        self.setWindowFlags(Qt.Tool)
        self.resize(300, 150)
        self.parent_ref = parent
        
        layout = QFormLayout()
        
        # Cỡ chữ
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(14, 60)
        self.slider_size.setValue(parent.font_size)
        self.slider_size.valueChanged.connect(self.update_font)
        layout.addRow("Cỡ chữ:", self.slider_size)

        # Độ mờ nền
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(int(parent.bg_opacity * 100))
        self.slider_opacity.valueChanged.connect(self.update_opacity)
        layout.addRow("Độ đậm nền:", self.slider_opacity)

        self.setLayout(layout)

    def update_font(self, val):
        self.parent_ref.font_size = val
        self.parent_ref.update_style()

    def update_opacity(self, val):
        self.parent_ref.bg_opacity = val / 100.0
        self.parent_ref.update_style()

# --- CỬA SỔ CHÍNH ---
class CaptionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Cấu hình mặc định
        self.font_size = 24
        self.bg_opacity = 0.7
        self.is_camera_on = False
        self.ws_connected = False

        # Thiết lập cửa sổ không viền, luôn ở trên cùng
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Kích thước & Vị trí ban đầu (Dưới đáy màn hình)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(100, screen.height() - 250, screen.width() - 200, 180)

        # --- GIAO DIỆN CHÍNH ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout chính
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. KHUNG CAMERA (Mặc định ẩn)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setSizePolicy(1, 1) # Expanding
        self.video_label.hide() # Ẩn lúc đầu
        self.main_layout.addWidget(self.video_label, 1) # Stretch factor 1

        # 2. KHUNG PHỤ ĐỀ (Container)
        self.text_container = QFrame()
        self.text_container.setObjectName("TextContainer")
        self.text_layout = QVBoxLayout(self.text_container)
        self.text_layout.setContentsMargins(10, 5, 10, 5)
        
        # Thanh công cụ nhỏ (Nút bấm)
        self.toolbar = QHBoxLayout()
        self.lbl_status = QLabel("🔴 Disconnected")
        self.lbl_status.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")
        
        self.btn_cam = QPushButton("📷 Bật Camera")
        self.btn_cam.setCursor(Qt.PointingHandCursor)
        self.btn_cam.clicked.connect(self.toggle_camera)
        
        self.btn_settings = QPushButton("⚙ Cài đặt")
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        self.btn_settings.clicked.connect(self.open_settings)
        
        self.btn_exit = QPushButton("❌ Thoát")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.clicked.connect(self.close_app)

        # Style cho nút bấm
        btn_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.4); }
        """
        self.btn_cam.setStyleSheet(btn_style)
        self.btn_settings.setStyleSheet(btn_style)
        self.btn_exit.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border-radius: 5px; }")

        self.toolbar.addWidget(self.lbl_status)
        self.toolbar.addStretch()
        self.toolbar.addWidget(self.btn_cam)
        self.toolbar.addWidget(self.btn_settings)
        self.toolbar.addWidget(self.btn_exit)
        
        self.text_layout.addLayout(self.toolbar)

        # Label Tiếng Việt
        self.lbl_viet = QLabel("Đang đợi kết nối máy chủ...")
        self.lbl_viet.setWordWrap(True)
        self.lbl_viet.setAlignment(Qt.AlignCenter)
        self.text_layout.addWidget(self.lbl_viet)

        # Label Tiếng Anh
        self.lbl_eng = QLabel("Waiting for server connection...")
        self.lbl_eng.setWordWrap(True)
        self.lbl_eng.setAlignment(Qt.AlignCenter)
        self.text_layout.addWidget(self.lbl_eng)

        self.main_layout.addWidget(self.text_container, 0) # Stretch factor 0 (giữ nguyên kích thước)

        # Nút kéo giãn (Resize Grip)
        self.sizegrip = QSizeGrip(self.central_widget)
        self.sizegrip.setStyleSheet("width: 20px; height: 20px; background-color: rgba(255, 255, 255, 0.3); border-top-left-radius: 10px;")

        # --- LOGIC WEBSOCKET (Chạy trên Main Thread để tránh lỗi QObject) ---
        self.ws = QWebSocket()
        self.ws.textMessageReceived.connect(self.on_ws_message)
        self.ws.connected.connect(self.on_ws_connected)
        self.ws.disconnected.connect(self.on_ws_disconnected)
        
        # Auto reconnect timer
        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.connect_ws)

        # Camera Thread
        self.thread_cam = VideoThread()
        self.thread_cam.change_pixmap_signal.connect(self.update_video_image)

        # Khởi chạy
        self.update_style()
        self.connect_ws()

        # Drag logic
        self.old_pos = self.pos()

    # --- XỬ LÝ GIAO DIỆN ---
    def update_style(self):
        bg_color = f"rgba(0, 0, 0, {int(self.bg_opacity * 255)})"
        self.text_container.setStyleSheet(f"""
            #TextContainer {{
                background-color: {bg_color};
                border-radius: 15px;
            }}
        """)
        self.lbl_viet.setStyleSheet(f"color: white; font-weight: bold; font-size: {self.font_size}px; font-family: Arial;")
        self.lbl_eng.setStyleSheet(f"color: #00FFFF; font-style: italic; font-size: {int(self.font_size * 0.75)}px; font-family: Arial;")

    def toggle_camera(self):
        if self.is_camera_on:
            # Tắt Camera -> Về chế độ Overlay gọn nhẹ
            self.video_label.hide()
            self.thread_cam.stop()
            self.btn_cam.setText("📷 Bật Camera")
            self.resize(self.width(), 180) # Thu nhỏ lại
        else:
            # Bật Camera -> Mở rộng cửa sổ
            self.video_label.show()
            self.thread_cam = VideoThread() # Re-init thread
            self.thread_cam.change_pixmap_signal.connect(self.update_video_image)
            self.thread_cam.start()
            self.btn_cam.setText("⏹ Tắt Camera")
            self.resize(self.width(), 600) # Phóng to ra
        
        self.is_camera_on = not self.is_camera_on

    def update_video_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec_()

    def close_app(self):
        if self.is_camera_on:
            self.thread_cam.stop()
        self.ws.close()
        self.close()

    # --- XỬ LÝ WEBSOCKET ---
    def connect_ws(self):
        if not self.ws_connected:
            self.lbl_status.setText("🟡 Connecting...")
            self.lbl_status.setStyleSheet("color: yellow;")
            self.ws.open(QUrl(f"ws://{config.HOST}:{config.PORT}/ws?role=viewer"))

    def on_ws_connected(self):
        self.ws_connected = True
        self.lbl_status.setText("🟢 Online")
        self.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.reconnect_timer.stop()
        self.lbl_viet.setText("Sẵn sàng...")
        self.lbl_eng.setText("Ready...")

    def on_ws_disconnected(self):
        self.ws_connected = False
        self.lbl_status.setText("🔴 Offline")
        self.lbl_status.setStyleSheet("color: red;")
        self.reconnect_timer.start(3000) # Thử lại sau 3s

    def on_ws_message(self, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "vad_start":
                self.lbl_status.setText("🎤 Listening...")
                self.lbl_status.setStyleSheet("color: #00FFFF; font-weight: bold;")
            elif msg_type == "vad_stop":
                self.lbl_status.setText("🟢 Online")
                self.lbl_status.setStyleSheet("color: #00FF00;")
            elif msg_type in ["realtime", "fullSentence"]:
                text = data.get("text", "")
                trans = data.get("trans", "")
                if text: self.lbl_viet.setText(text)
                if trans: self.lbl_eng.setText(trans)
        except:
            pass

    # --- KÉO THẢ CỬA SỔ ---
    def resizeEvent(self, event):
        # Luôn giữ SizeGrip ở góc dưới phải
        rect = self.rect()
        self.sizegrip.move(rect.right() - self.sizegrip.width(), rect.bottom() - self.sizegrip.height())
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptionWindow()
    window.show()
    sys.exit(app.exec_())
