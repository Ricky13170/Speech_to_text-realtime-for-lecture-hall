# Chạy luồng OpenCV song song khi server start
# threading.Thread(target=overlay_thread_func, daemon=True).start()
import cv2
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

overlay_text = {
    "partial": "",
    "final": "",
    "vad": "",
}
lock_overlay = threading.Lock()

WINDOW_NAME = "ASR Caption Overlay"

def draw_text_box(draw, text, xy, font, box_color=(0, 0, 0, 150), text_color=(255,255,255)):
    """
    Vẽ text có hộp nền bán trong suốt → dễ đọc hơn
    """
    if not text:
        return

    x, y = xy
    w, h = draw.textsize(text, font=font)

    # Khung bán trong suốt
    draw.rectangle(
        (x-10, y-5, x + w + 10, y + h + 5),
        fill=box_color
    )
    draw.text((x, y), text, font=font, fill=text_color)

def update_overlay_text(t_type, text):
    with lock_overlay:
        if t_type == "realtime":
            overlay_text["partial"] = text
        elif t_type == "fullSentence":
            overlay_text["final"] = text
            overlay_text["partial"] = ""
        elif t_type == "vad_start":
            overlay_text["vad"] = "🎤 Listening..."
        elif t_type == "vad_stop":
            overlay_text["vad"] = ""

def overlay_thread_func():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Webcam không mở được → dùng nền đen")
        cap = None

    FONT_PATH = "C:/Windows/Fonts/arial.ttf"
    font_vad     = ImageFont.truetype(FONT_PATH, 28)
    font_partial = ImageFont.truetype(FONT_PATH, 32)
    font_final   = ImageFont.truetype(FONT_PATH, 36)

    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = np.zeros((480, 800, 3), dtype=np.uint8)

        with lock_overlay:
            vad_msg = overlay_text["vad"]
            partial = overlay_text["partial"]
            final   = overlay_text["final"]

        # Convert sang PIL để vẽ đẹp
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, "RGBA")

        y = 20

        # --- VAD STATUS ---
        if vad_msg:
            draw_text_box(draw, vad_msg, (20, y), font_vad, box_color=(255,255,0,120))
            y += 45

        # --- PARTIAL (Realtime) ---
        if partial:
            draw_text_box(draw, partial, (20, y), font_partial, box_color=(0,0,0,150))
            y += 50

        # --- FINAL ---
        if final:
            draw_text_box(draw, final, (20, y), font_final, box_color=(0,80,0,160), text_color=(200,255,200))

        # Convert lại CV2
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

# chạy thread overlay
threading.Thread(target=overlay_thread_func, daemon=True).start()
