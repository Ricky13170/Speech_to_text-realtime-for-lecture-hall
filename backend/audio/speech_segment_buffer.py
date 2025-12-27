import numpy as np

class SpeechSegmentBuffer:
    def __init__(self, sample_rate, max_sec, overlap_sec, silence_limit):
        self.sr = sample_rate
        self.max_sec = max_sec
        self.overlap_sec = overlap_sec
        self.silence_limit = silence_limit
        self.reset()

    def reset(self):
        self.in_speech = False
        self.segment = []
        self.overlap = np.zeros(0, np.float32)
        self.last_voice_ts = 0.0

    def process(self, audio, is_speech, now_ts):
        """
        Returns:
        - None
        - ("partial", audio_chunk)
        - ("final", audio_chunk)
        """
        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.segment = [self.overlap] if len(self.overlap) else []
            self.last_voice_ts = now_ts
            self.segment.append(audio)
            return None

        if not self.in_speech:
            return None

        silence = now_ts - self.last_voice_ts
        self.segment.append(audio)

        total_sec = len(np.concatenate(self.segment)) / self.sr

        if silence >= self.silence_limit:
            chunk = np.concatenate(self.segment)
            self._update_overlap(chunk)
            self.reset()
            return "final", chunk

        if total_sec >= self.max_sec:
            chunk = np.concatenate(self.segment)
            self._update_overlap(chunk)
            self.segment = [self.overlap]
            return "partial", chunk

        return None

    def _update_overlap(self, chunk):
        n = int(self.overlap_sec * self.sr)
        self.overlap = chunk[-n:] if len(chunk) >= n else chunk
