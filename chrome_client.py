import asyncio
import json
import struct
import websockets
import numpy as np
import sounddevice as sd
import argparse

class ChromeASRClient:
    def __init__(self, bridge_url="ws://localhost:8765", asr_url=None):
        self.bridge_url = bridge_url
        self.asr_url = asr_url
        
    async def connect_to_bridge(self):
        """Kết nối đến Chrome Audio Bridge"""
        print(f"Connecting to Chrome Audio Bridge: {self.bridge_url}")
        
        async with websockets.connect(self.bridge_url) as ws:
            # Gửi lệnh bắt đầu
            await ws.send(json.dumps({"action": "start_forwarding"}))
            print("Connected to bridge. Audio will be forwarded to ASR.")
            
            # Nhận status updates
            async for message in ws:
                data = json.loads(message)
                print(f"Bridge status: {data}")
                
    async def direct_to_asr(self):
        """Kết nối trực tiếp đến ASR (nếu không dùng bridge)"""
        if not self.asr_url:
            print("Vui lòng cung cấp ASR URL")
            return
            
        print(f"Connecting directly to ASR: {self.asr_url}")
        
        async with websockets.connect(self.asr_url, max_size=None) as ws:
            print("Connected to ASR server")
            
            # Nhận transcriptions
            async for message in ws:
                data = json.loads(message)
                t = data.get("type")
                
                if t == "realtime":
                    print(f"\r[...] {data.get('text','')}", end="", flush=True)
                elif t == "fullSentence":
                    print("\r" + " " * 80, end="")
                    print(f"\r[VI] {data.get('text','')}")
                    print(f"[EN] {data.get('trans','')}\n")
                    
    def run(self):
        asyncio.run(self.connect_to_bridge())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", default="ws://localhost:8765", help="Chrome Bridge WebSocket URL")
    parser.add_argument("--asr", help="ASR server WebSocket URL")
    
    args = parser.parse_args()
    
    client = ChromeASRClient(
        bridge_url=args.bridge,
        asr_url=args.asr
    )
    client.run()
