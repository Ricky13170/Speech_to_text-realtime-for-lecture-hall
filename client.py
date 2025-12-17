import asyncio
import json
import queue
import struct
import sys
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

import config

# =========================
# AUDIO QUEUE
# =========================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio: {status}", file=sys.stderr)
    audio_queue.put(np.squeeze(indata).copy())


# =========================
# ENERGY GATE CONFIG
# =========================
ENERGY_THRESHOLD = 0.01        # 🔧 Tune theo mic (0.005 – 0.02)
SILENCE_HYSTERESIS = 5         # số frame silence để tắt speech
PREROLL_FRAMES = 2             # gửi bù để không mất phụ âm đầu


def rms_energy(x: np.ndarray) -> float:
    """Root Mean Square energy"""
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x ** 2)))


# =========================
# MAIN
# =========================
async def main():
    viewer_url = (
        config.MODAL_URL.replace("wss://", "https://") + "/viewer"
        if config.USE_MODAL
        else f"http://127.0.0.1:{config.PORT}"
    )

    print("=" * 60)
    print("ASR Realtime Client (Energy-gated)")
    print("=" * 60)
    print(f"Server: {config.WS_URL}")
    print(f"Viewer: {viewer_url}")
    print("=" * 60)
    print("Open the viewer URL in browser to see transcription")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    print("Connecting...")
    async with websockets.connect(config.WS_URL, max_size=None) as ws:
        print("Connected! Start speaking...\n")

        # =========================
        # AUDIO STREAM
        # =========================
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=config.CHUNK_SAMPLES,
        )
        stream.start()

        # =========================
        # RECEIVE TASK
        # =========================
        async def recv():
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")

                if t == "realtime":
                    text = data.get("text", "")
                    if text:
                        print(f"\r[...] {text}", end="", flush=True)

                elif t == "fullSentence":
                    print("\r" + " " * 80, end="")
                    print(f"\r[VI] {data.get('text','')}")
                    print(f"[EN] {data.get('trans','')}\n")

                elif t == "vad_start":
                    print("\n[Listening...]")

        recv_task = asyncio.create_task(recv())

        # =========================
        # SEND LOOP (ENERGY GATE)
        # =========================
        meta = json.dumps({"sampleRate": config.SAMPLE_RATE}).encode()
        header = struct.pack("<I", len(meta)) + meta

        speech_active = False
        silence_frames = 0
        preroll = deque(maxlen=PREROLL_FRAMES)

        try:
            while True:
                if audio_queue.empty():
                    await asyncio.sleep(0.005)
                    continue

                chunk = audio_queue.get()

                # 🔊 Energy measurement
                energy = rms_energy(chunk)

                if energy >= ENERGY_THRESHOLD:
                    if not speech_active:
                        speech_active = True
                        silence_frames = 0
                    else:
                        silence_frames = 0
                else:
                    if speech_active:
                        silence_frames += 1
                        if silence_frames >= SILENCE_HYSTERESIS:
                            speech_active = False

                # Lưu preroll
                preroll.append(chunk)

                # ⛔ Không gửi nếu không có giọng nói
                if not speech_active:
                    continue

                # 🚀 Gửi preroll khi vừa bật speech
                while preroll:
                    pcm = preroll.popleft().astype(np.int16).tobytes()
                    await ws.send(header + pcm)

        except (KeyboardInterrupt, websockets.exceptions.ConnectionClosed):
            print("\nDisconnected")

        finally:
            stream.stop()
            recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
