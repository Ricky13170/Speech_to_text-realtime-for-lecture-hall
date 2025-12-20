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

# ================= AUDIO QUEUE =================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(np.squeeze(indata).copy())

# ================= ENERGY GATE =================
ENERGY_THRESHOLD = 0.03
SILENCE_FRAMES = 5
PREROLL_FRAMES = 2

def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

# ================= MAIN =================
async def main():
    viewer_url = config.MODAL_URL.replace("wss://", "https://") + "/viewer"

    print("=" * 60)
    print("ASR Realtime Client (Energy-gated)")
    print("=" * 60)
    print(f"Server: {config.WS_URL}")
    print(f"Viewer: {viewer_url}")
    print("=" * 60)

    async with websockets.connect(config.WS_URL, max_size=None) as ws:
        print("Connected! Start speaking...\n")

        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=config.CHUNK_SAMPLES,
        )
        stream.start()

        async def recv():
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")

                if t == "realtime":
                    print(f"\r[...] {data.get('text','')}", end="", flush=True)

                elif t == "fullSentence":
                    print("\r" + " " * 80, end="")
                    print(f"\r[VI] {data.get('text','')}")
                    print(f"[EN] {data.get('trans','')}\n")

                elif t == "vad_start":
                    print("\n[Listening...]")

        recv_task = asyncio.create_task(recv())

        meta = json.dumps({"sr": config.SAMPLE_RATE}).encode()
        header = struct.pack("<I", len(meta)) + meta

        speech = False
        silence = 0
        preroll = deque(maxlen=PREROLL_FRAMES)

        try:
            while True:
                if audio_queue.empty():
                    await asyncio.sleep(0.005)
                    continue

                chunk = audio_queue.get()
                e = rms(chunk)
                preroll.append(chunk)

                if e >= ENERGY_THRESHOLD:
                    speech = True
                    silence = 0
                elif speech:
                    silence += 1
                    if silence >= SILENCE_FRAMES:
                        speech = False

                if not speech:
                    continue

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
