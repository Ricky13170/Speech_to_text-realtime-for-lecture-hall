import sounddevice as sd
import numpy as np
import queue
import struct
import json
import sys
import asyncio
import websockets
import config

# --- STATE ---
BUFFER = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    mono = np.squeeze(indata)
    BUFFER.put(mono.copy())

async def send_loop():
    uri = config.WS_URL
    async with websockets.connect(uri, max_size=None) as ws:
        print(f"Connected to server: {uri}")

        # Start audio stream
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, 
            channels=1, 
            dtype='int16', 
            callback=audio_callback, 
            blocksize=config.CHUNK_SAMPLES
        )
        stream.start()

        # Task to receive messages from server
        async def recv_loop():
            async for msg in ws:
                data = json.loads(msg)
                ttype = data.get("type")
                
                if ttype == "realtime":
                    print(f"[Partial] {data.get('text')} | {data.get('trans')}")
                elif ttype == "fullSentence":
                    print(f"[Final] {data.get('text')}")
                    print(f"   -> {data.get('trans')}")
                elif ttype == "vad_start":
                    print("[VAD] Start speaking")
                elif ttype == "vad_stop":
                    print("[VAD] Stop speaking")
                else:
                    print(f"[Server] {data}")

        recv_task = asyncio.create_task(recv_loop())

        try:
            while True:
                if BUFFER.empty():
                    await asyncio.sleep(0.01)
                    continue
                
                chunk = BUFFER.get()
                pcm_bytes = chunk.astype(np.int16).tobytes()
                
                metadata = {"sampleRate": config.SAMPLE_RATE}
                meta_json = json.dumps(metadata)
                meta_len = len(meta_json)
                
                message = struct.pack("<I", meta_len) + meta_json.encode("utf-8") + pcm_bytes
                
                try:
                    await ws.send(message)
                except websockets.exceptions.ConnectionClosed:
                    print("\n⚠️ Server disconnected. Stopping client.")
                    break

        except KeyboardInterrupt:
            print("\nStopping client...")
        finally:
            stream.stop()
            recv_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(send_loop())
    except KeyboardInterrupt:
        pass
