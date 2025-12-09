import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import sys
import argparse
import struct
from config import *

# --- CẤU HÌNH ---
parser = argparse.ArgumentParser()
parser.add_argument("--loopback", action="store_true", help="Chế độ thu âm thanh từ máy tính (YouTube/Video)")
args = parser.parse_args()

def get_wasapi_loopback_device():
    """
    Tìm thiết bị Loopback của Windows (WASAPI).
    Giúp thu âm thanh đang phát ra loa mà không cần Micro.
    """
    print("\n[INFO] Đang tìm thiết bị âm thanh hệ thống (Loopback)...")
    
    try:
        # 1. Tìm Host API là WASAPI
        host_apis = sd.query_hostapis()
        wasapi_index = next((i for i, api in enumerate(host_apis) if 'WASAPI' in api['name']), None)
        
        if wasapi_index is None:
            print("[ERR] Máy bạn không hỗ trợ WASAPI. Hãy dùng Mic thường.")
            return None

        # 2. Tìm thiết bị Loopback trong danh sách WASAPI
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            # Chỉ xét thiết bị thuộc WASAPI
            if dev['hostapi'] == wasapi_index:
                # Thiết bị Loopback thường có tên chứa "loopback" (trên Windows mới)
                # Hoặc nó là thiết bị Input nhưng lại liên kết với Speakers
                if "loopback" in dev['name'].lower():
                    print(f"[OK] Đã chọn: {dev['name']} (ID: {i})")
                    return i
        
        # Nếu không tìm thấy chữ "loopback", thử tìm thiết bị Output mặc định rồi dùng loopback của nó
        # Trên WASAPI, thiết bị loopback thường nằm ngay sau thiết bị output hoặc có ID riêng
        # Đây là fallback đơn giản:
        print("[WARN] Không tìm thấy thiết bị tên 'Loopback'. Đang thử dùng thiết bị mặc định của WASAPI.")
        return None 

    except Exception as e:
        print(f"[ERR] Lỗi tìm thiết bị: {e}")
        return None

async def send_audio():
    # Chọn thiết bị: Nếu có cờ --loopback thì tìm Loopback, không thì dùng Mic mặc định
    device_id = get_wasapi_loopback_device() if args.loopback else None
    
    # Nếu dùng Loopback, ta cần biết Sample Rate gốc của loa (thường là 48000 hoặc 44100)
    # sounddevice sẽ tự convert nếu phần cứng hỗ trợ, nhưng tốt nhất là lấy info
    if device_id is not None:
        try:
            dev_info = sd.query_devices(device_id)
            print(f"[INFO] Thiết bị đang chạy ở tần số: {dev_info['default_samplerate']}Hz")
        except Exception:
            pass
    
    print(f"\nConnecting to {WS_URL}...")
    print(f"Mode: {'SYSTEM AUDIO (YouTube)' if args.loopback else 'MICROPHONE'}")
    print("Press Ctrl+C to stop.\n")

    while True: # Reconnect loop
        try:
            async with websockets.connect(WS_URL) as websocket:
                print("✅ Connected to Server.")
                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()

                def callback(indata, frames, time, status):
                    if status:
                        pass 
                    loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

                # Mở stream
                try:
                    stream = sd.InputStream(
                        device=device_id,
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        blocksize=CHUNK_SAMPLES,
                        callback=callback,
                        dtype="int16"
                    )
                    
                    with stream:
                        print("🔴 Đang lắng nghe... (Hãy bật video YouTube lên)")
                        
                        while True:
                            data = await queue.get()
                            
                            metadata = {"sampleRate": SAMPLE_RATE}
                            meta_json = json.dumps(metadata)
                            meta_len = len(meta_json)
                            pcm_bytes = data.tobytes()
                            
                            message = struct.pack("<I", meta_len) + meta_json.encode("utf-8") + pcm_bytes
                            
                            await websocket.send(message)
                            
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                                res_json = json.loads(response)
                                
                                if res_json.get("type") == "realtime":
                                    # Fix lỗi in ký tự lạ trên Windows bằng cách encode/decode
                                    text = res_json.get('text', '').encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                                    print(f"\r[Đang nghe]: {text}", end="", flush=True)
                                elif res_json.get("type") == "fullSentence":
                                    text = res_json.get('text', '').encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                                    trans = res_json.get('trans', '').encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                                    print(f"\r[Kết quả]: {text} \n   -> {trans}\n")
                                elif res_json.get("type") == "vad_start":
                                     print("\r[VAD] ...Speaking...", end="", flush=True)
                                    
                            except asyncio.TimeoutError:
                                pass
                            except websockets.exceptions.ConnectionClosed:
                                print("\n⚠️ Server disconnected.")
                                break

                except Exception as e:
                    print(f"\n[ERROR] Stream error: {e}")
                    break
        
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
            print("\n⏳ Mất kết nối. Đang thử lại sau 3s...", end="", flush=True)
            await asyncio.sleep(3)
        except Exception as e:
            print(f"\n[FATAL] Lỗi không xác định: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(send_audio())
    except KeyboardInterrupt:
        print("\nStopped.")
