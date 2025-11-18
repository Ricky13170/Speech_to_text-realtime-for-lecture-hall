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
        print("⚠️ Không mở được webcam, chuyển sang nền đen")
        cap = None

    FONT_PATH = "C:/Windows/Fonts/arial.ttf"  # hoặc "Roboto-Regular.ttf"
    font_vad = ImageFont.truetype(FONT_PATH, 28)
    font_partial = ImageFont.truetype(FONT_PATH, 30)
    font_final = ImageFont.truetype(FONT_PATH, 34)

    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with lock_overlay:
            vad_msg = overlay_text["vad"]
            partial = overlay_text["partial"]
            final = overlay_text["final"]

        # Chuyển frame sang định dạng Pillow
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        y = 20
        if vad_msg:
            draw.text((20, y), vad_msg, font=font_vad, fill=(255, 255, 0))
            y += 40
        if partial:
            draw.text((20, y), partial, font=font_partial, fill=(255, 255, 255))
            y += 40
        if final:
            draw.text((20, y), final, font=font_final, fill=(0, 255, 0))

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("ASR Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    if cap:
        cap.release()
    cv2.destroyAllWindows()


# Chạy luồng OpenCV song song khi server start
threading.Thread(target=overlay_thread_func, daemon=True).start()
