import json
import base64
import asyncio
import numpy as np
import time

from backend.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, ASR_DEVICE,
    NLLB_MODEL, NLLB_SRC_LANG, NLLB_TGT_LANG, NLLB_DEVICE,
    VAD_THRESHOLD, MIN_SILENCE_DURATION, MAX_BUFFER_DURATION
)
from backend.asr import WhisperASR, StreamingASRProcessor
from backend.translation import NLLBTranslator


class ASRService:
    def __init__(self):
        self.asr = None
        self.translator = None
        self.is_initialized = False

    async def init(self):
        print("[Model] Loading Whisper...")
        
        self.asr = WhisperASR(
            model_size=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
            device=ASR_DEVICE,
            cache_dir="/cache/whisper"
        )
        self.asr.load_model()
        
        print("[Model] Loading NLLB Translator...")
        loop = asyncio.get_event_loop()
        self.translator = NLLBTranslator(
            model_name=NLLB_MODEL, 
            src_lang=NLLB_SRC_LANG, 
            tgt_lang=NLLB_TGT_LANG, 
            device=NLLB_DEVICE, 
            cache_dir="/cache/nllb"
        )
        await loop.run_in_executor(None, self.translator.load_model)
        
        self.is_initialized = True
        print("[Model] All models ready")

    def create_session(self):
        return ASRSession(self)


class ASRSession:
    def __init__(self, service: ASRService):
        self.service = service
        self.out_queue = asyncio.Queue()
        self.processor = StreamingASRProcessor(service.asr, context_size=3)
        self.is_recording = False
        self.last_partial_time = 0
        self.silence_counter = 0.0
        self.last_partial_text = ""
        self.lock = asyncio.Lock()
        self.segment_id = 0
        self.buffer_start_time = 0
        
        # Language config
        self.src_lang = None
        self.tgt_lang = 'en'
        self.do_translate = True
        
        # Context memory
        self.context_keywords = []
        self.transcript_history = []

    async def handle_incoming(self, message: str):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "start":
                await self._start(data)
            elif msg_type == "audio":
                await self._audio(data)
            elif msg_type == "stop":
                await self._stop()
            elif msg_type == "context":
                self._set_context(data)
            elif msg_type == "ping":
                await self.out_queue.put(json.dumps({"type": "pong"}))
        except Exception as e:
            print(f"[Handler] Error: {e}")

    def _set_context(self, data):
        keywords = data.get("keywords", [])
        context = data.get("context", "")
        self.context_keywords = keywords if isinstance(keywords, list) else []
        if context:
            self.context_keywords.append(context)
        print(f"[Context] Keywords: {self.context_keywords}")

    def _build_prompt(self):
        parts = []
        if self.context_keywords:
            parts.append(" ".join(self.context_keywords[:10]))
        if self.transcript_history:
            parts.append(" ".join(self.transcript_history[-2:]))
        return " ".join(parts) if parts else None

    async def _start(self, data):
        self.src_lang = data.get("srcLang")
        self.tgt_lang = data.get("tgtLang")
        self.do_translate = data.get("translate", True)
        
        if self.src_lang:
            self.service.asr.language = self.src_lang
        else:
            self.service.asr.language = None
        
        if self.do_translate and self.src_lang and self.tgt_lang:
            lang_map = {'vi': 'vie_Latn', 'en': 'eng_Latn'}
            self.service.translator.src_lang = lang_map.get(self.src_lang, 'vie_Latn')
            self.service.translator.tgt_lang = lang_map.get(self.tgt_lang, 'eng_Latn')
        
        self.processor.init()
        self.is_recording = True
        self.silence_counter = 0.0
        self.last_partial_text = ""
        self.segment_id = 0
        self.buffer_start_time = time.time()
        self.transcript_history = []
        
        mode = f"{self.src_lang or 'auto'}"
        if self.do_translate:
            mode += f" → {self.tgt_lang}"
        print(f"[Session] Started ({mode})")
        await self.out_queue.put(json.dumps({"type": "status", "status": "started"}))

    async def _audio(self, data):
        if not self.is_recording:
            return
            
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
        except Exception:
            return

        async with self.lock:
            await self._process(audio)

    async def _process(self, audio: np.ndarray):
        # Energy-based VAD
        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0
        is_speech = rms > VAD_THRESHOLD
        
        chunk_duration = len(audio) / 16000
        buffer_duration = len(self.processor.audio_buffer) / 16000
        
        if is_speech:
            self.silence_counter = 0.0
            self.processor.insert_audio(audio)
        else:
            self.silence_counter += chunk_duration
            if self.silence_counter < MIN_SILENCE_DURATION:
                self.processor.insert_audio(audio)

        loop = asyncio.get_event_loop()
        now = time.time()
        
        should_finalize = False
        reason = ""
        
        if self.silence_counter > MIN_SILENCE_DURATION and buffer_duration > 0.5:
            should_finalize = True
            reason = f"silence {self.silence_counter:.1f}s"
        elif buffer_duration > MAX_BUFFER_DURATION:
            should_finalize = True
            reason = f"max {buffer_duration:.1f}s"
        
        if should_finalize and len(self.processor.audio_buffer) > 0:
            print(f"[VAD] Finalize ({reason})")
            
            text, conf = await loop.run_in_executor(None, self.processor.process_final)
            if text:
                self.segment_id += 1
                self.transcript_history.append(text)
                if len(self.transcript_history) > 5:
                    self.transcript_history.pop(0)
                
                print(f"[ASR] #{self.segment_id}: {text[:80]}...")
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id,
                    "source": text,
                    "target": "", 
                    "is_final": True,
                    "confidence": conf
                }))
                
                if self.do_translate:
                    asyncio.create_task(self._translate(text, self.segment_id))
            
            self.silence_counter = 0.0
            self.last_partial_text = ""
            self.buffer_start_time = time.time()
            
        elif is_speech and (now - self.last_partial_time > 1.5) and buffer_duration > 1.5:
            text, conf = await loop.run_in_executor(None, self.processor.process_partial)
            
            if text and len(text) > len(self.last_partial_text) + 10:
                self.last_partial_text = text
                self.last_partial_time = now
                
                display = text[:60] + "..." if len(text) > 60 else text
                print(f"[Partial] {display}")
                
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id + 1,
                    "source": text,
                    "target": "",
                    "is_final": False,
                    "confidence": conf
                }))

    async def _translate(self, text: str, seg_id: int):
        loop = asyncio.get_event_loop()
        try:
            start = time.time()
            translated = await loop.run_in_executor(None, self.service.translator.translate, text)
            duration = time.time() - start
            if translated:
                print(f"[MT] #{seg_id} ({duration:.1f}s): {translated[:60]}...")
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": seg_id,
                    "source": text,
                    "target": translated,
                    "is_final": True,
                    "confidence": 1.0 
                }))
        except Exception as e:
            print(f"[MT] Error: {e}")

    async def _stop(self):
        self.is_recording = False
        loop = asyncio.get_event_loop()
        text, conf = await loop.run_in_executor(None, self.processor.process_final)
        
        if text:
            self.segment_id += 1
            self.transcript_history.append(text)
            print(f"[ASR] #{self.segment_id}: {text[:80]}...")
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": self.segment_id,
                "source": text,
                "target": "",
                "is_final": True,
                "confidence": conf
            }))
            if self.do_translate:
                await self._translate(text, self.segment_id)
        
        print("[Session] Stopped")
        await self.out_queue.put(json.dumps({"type": "status", "status": "stopped"}))
