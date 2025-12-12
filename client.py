import asyncio
import json
import queue
import struct
import sys

import numpy as np
import sounddevice as sd
import websockets

import config

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio: {status}", file=sys.stderr)
    audio_queue.put(np.squeeze(indata).copy())


async def main():
    viewer_url = config.MODAL_URL.replace("wss://", "https://") + "/viewer" if config.USE_MODAL else f"http://127.0.0.1:{config.PORT}"
    
    print("=" * 60)
    print("ASR Realtime Client")
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

        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype='int16',
            callback=audio_callback,
            blocksize=config.CHUNK_SAMPLES
        )
        stream.start()

        async def recv():
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "realtime":
                    text = data.get('text', '')
                    trans = data.get('trans', '')
                    if text:
                        print(f"\r[...] {text}", end="", flush=True)
                elif t == "fullSentence":
                    text = data.get('text', '')
                    trans = data.get('trans', '')
                    print(f"\r[VI] {text}")
                    print(f"[EN] {trans}\n")
                elif t == "vad_start":
                    print("\n[Listening...]")
                elif t == "vad_stop":
                    pass

        recv_task = asyncio.create_task(recv())

        try:
            meta = json.dumps({"sampleRate": config.SAMPLE_RATE}).encode()
            header = struct.pack("<I", len(meta)) + meta

            while True:
                if audio_queue.empty():
                    await asyncio.sleep(0.01)
                    continue

                chunk = audio_queue.get()
                pcm = chunk.astype(np.int16).tobytes()
                await ws.send(header + pcm)

        except (KeyboardInterrupt, websockets.exceptions.ConnectionClosed):
            print("\nDisconnected")
        finally:
            stream.stop()
            recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
