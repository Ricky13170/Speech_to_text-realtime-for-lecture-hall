import json
import base64
import asyncio
import numpy as np
import time
from datetime import datetime

from backend.config import (
    SAMPLE_RATE, MAX_SEGMENT_SEC, OVERLAP_SEC, SILENCE_LIMIT,
    MIN_DECODE_SEC, AGREEMENT_N, HALLUCINATION_HISTORY_SIZE
)
from backend.asr import WhisperASR, StreamingASRProcessor
from backend.translation import NLLBTranslator
from backend.vad import VADManager
from backend.speech_segment_buffer import SpeechSegmentBuffer
from backend.local_agreement import LocalAgreement
from backend.hallucination_filter import HallucinationFilter


class ASRService:
    """Service manages shared models across sessions"""
    
    def __init__(self):
        self.asr = None
        self.translator = None
        self.is_initialized = False

    async def init(self):
        print("[Model] Loading Whisper...")
        
        self.asr = WhisperASR(
            model_size="large-v3",
            language="vi",
            device="cuda",
            cache_dir="/cache/whisper"
        )
        self.asr.load_model()
        
        print("[Model] Loading NLLB Translator...")
        loop = asyncio.get_event_loop()
        self.translator = NLLBTranslator(
            model_name="facebook/nllb-200-3.3B",
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
            device="cuda",
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
        
        # Core components
        self.vad = VADManager()
        self.segmenter = SpeechSegmentBuffer(
            sample_rate=SAMPLE_RATE,
            max_sec=MAX_SEGMENT_SEC,
            overlap_sec=OVERLAP_SEC,
            silence_limit=SILENCE_LIMIT
        )
        self.agreement = LocalAgreement(n=AGREEMENT_N)
        self.hallucination_filter = HallucinationFilter(
            history_size=HALLUCINATION_HISTORY_SIZE
        )
        
        # Session state
        self.is_recording = False
        self.session_start = None
        self.segment_id = 0
        self.last_stable = ""
        self.pending_partial_text = None      
        self.pending_partial_time = None      
        self.PARTIAL_FINALIZE_TIMEOUT = 3.0 
        
        # Language config
        self.do_translate = True
        
        self.transcript_history = [] 
        
        print("[Session] Initialized with segment-based architecture")

    async def handle_incoming(self, message: str):
        """Route incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "start":
                await self._handle_start(data)
            elif msg_type == "audio":
                await self._handle_audio(data)
            elif msg_type == "stop":
                await self._handle_stop()
            elif msg_type == "ping":
                await self.out_queue.put(json.dumps({"type": "pong"}))
                
        except Exception as e:
            print(f"[Handler] Error: {e}")

    async def _handle_start(self, data):
        """Start recording session"""
        print("[Session] START")
        
        self.is_recording = True
        self.session_start = datetime.now()
        self.segment_id = 0
        self.last_stable = ""
        
        # Reset all components
        self.vad.reset()
        self.segmenter.reset()
        self.agreement.reset()
        self.hallucination_filter.reset()
        self.transcript_history = []  
        
        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "started"
        }))

    async def _handle_audio(self, data):
        """Process incoming audio chunk"""
        if not self.is_recording:
            return
        
        if self.pending_partial_text and self.pending_partial_time:
            if time.time() - self.pending_partial_time > self.PARTIAL_FINALIZE_TIMEOUT:
                await self._finalize_pending_partial()

        # Decode audio
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[Audio] Decode error: {e}")
            return
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.003:  
            return

        is_speech = self.vad.is_speech(audio)
        

        now_ts = (datetime.now() - self.session_start).total_seconds()
        result = self.segmenter.process(audio, is_speech, now_ts)
        
        if result is None:

            buffer_duration = self.segmenter.get_current_duration()
            
            if (buffer_duration >= 2.0 and 
                now_ts - getattr(self, '_last_intermediate_decode', 0) > 1.5):
                
                self._last_intermediate_decode = now_ts
                current_audio = self.segmenter.get_current_audio()
                
                if len(current_audio) > 16000:
                    asyncio.create_task(
                        self._decode_intermediate(current_audio)
                    )
            
            return 
        
        kind, chunk = result
        duration = len(chunk) / SAMPLE_RATE
        
        print(f"\n🧩 Segment → {kind.upper()} | {duration:.2f}s")
        
        await self._decode_segment(chunk, is_final=(kind == "final"))

    async def _decode_segment(self, audio: np.ndarray, is_final: bool):
        """Decode audio segment with hallucination filtering and local agreement"""
        
        duration = len(audio) / SAMPLE_RATE
        
        if not is_final and duration < MIN_DECODE_SEC:
            print("⚠️ Skip PARTIAL (too short)")
            return
        
        loop = asyncio.get_event_loop()
        start = time.time()
        
        prompt = self._build_prompt()
        
        text, conf = await loop.run_in_executor(
            None,
            self.service.asr.transcribe,
            audio,
            prompt 
        )
        
        inference_ms = int((time.time() - start) * 1000)
        
        if not text:
            return

        audio_rms = np.sqrt(np.mean(audio ** 2))
        is_hallu, reason = self.hallucination_filter.is_hallucination(
            text, audio_rms, conf
        )
        
        if is_hallu:
            print(f"⚠️ Hallucination filtered: {reason}")
            return
        
        print(f"\n📝 {'FINAL' if is_final else 'PARTIAL'} [VI] {text}")
        
        if is_final:

            if self.pending_partial_text:
                await self._finalize_pending_partial()

            stable_text = text
            unstable_text = ""
            self.segment_id += 1
            self.last_stable = stable_text
            self.agreement.reset()
            
            self.transcript_history.append(stable_text)
            if len(self.transcript_history) > 3:
                self.transcript_history.pop(0)
        else:
            # PARTIAL: Apply local agreement
            stable_text, unstable_text = self.agreement.process(text)
            
            self.pending_partial_text = f"{stable_text} {unstable_text}".strip()
            self.pending_partial_time = time.time()

        # Combine for display
        display_text = f"{stable_text} {unstable_text}".strip()
        
        await self.out_queue.put(json.dumps({
            "type": "transcript",
            "segment_id": self.segment_id + (0 if is_final else 1),
            "source": display_text,
            "target": "",
            "is_final": is_final,
            "confidence": conf,
            "processing_ms": inference_ms,
            "timestamp": self._get_timestamp()
        }))
        
        if is_final and self.do_translate and stable_text:
            asyncio.create_task(self._translate(stable_text, self.segment_id))

    async def _translate(self, text: str, seg_id: int):
        """Translate Vietnamese to English"""
        loop = asyncio.get_event_loop()
        
        try:
            start = time.time()
            translated = await loop.run_in_executor(
                None,
                self.service.translator.translate,
                text
            )
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

    async def _handle_stop(self):
        """Stop recording session"""
        print("[Session] STOP")
        
        self.is_recording = False
        
        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "stopped"
        }))

    async def _finalize_pending_partial(self):
        """Auto-finalize pending partial after timeout"""
        if not self.pending_partial_text:
            return
    
        print(f"⏰ Auto-finalize pending partial: {self.pending_partial_text[:50]}...")
    
        self.segment_id += 1
        self.last_stable = self.pending_partial_text
    
        await self.out_queue.put(json.dumps({
            "type": "transcript",
            "segment_id": self.segment_id,
            "source": self.pending_partial_text,
            "target": "",
            "is_final": True,  
            "confidence": 0.8,
            "timestamp": self._get_timestamp()
        }))
    
        # Trigger translation
        if self.do_translate:
            asyncio.create_task(self._translate(self.pending_partial_text, self.segment_id))
    
        # Clear pending
        self.pending_partial_text = None
        self.pending_partial_time = None
        self.agreement.reset()

    def _get_timestamp(self) -> str:
        """Get current session timestamp in MM:SS format"""
        if not self.session_start:
            return "00:00"
        
        elapsed = (datetime.now() - self.session_start).total_seconds()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _build_prompt(self) -> str:
        """
        Build Whisper prompt from recent history.
        This significantly improves accuracy by providing context.
        """
        if not self.transcript_history:
            return None
        
        # Use last 2-3 transcripts (max ~100 chars)
        recent = " ".join(self.transcript_history[-2:])
        
        if len(recent) > 150:
            recent = recent[-150:]
        
        return recent
    
    async def _decode_intermediate(self, audio: np.ndarray):
        """
        Intermediate partial decode for live feedback.
        Less strict filtering than regular partials.
        """
        loop = asyncio.get_event_loop()
        
        try:
            prompt = self._build_prompt()
            text, conf = await loop.run_in_executor(
                None,
                self.service.asr.transcribe,
                audio,
                prompt
            )
            
            if text and len(text) > 10: 
                # Apply local agreement
                stable_text, unstable_text = self.agreement.process(text)
                display_text = f"{stable_text} {unstable_text}".strip()
                
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id + 1,
                    "source": display_text,
                    "target": "",
                    "is_final": False,
                    "confidence": conf,
                    "timestamp": self._get_timestamp()
                }))
        except Exception as e:
            print(f"[Intermediate] Error: {e}")