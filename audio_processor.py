import torch
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import config

print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    force_reload=False,
    trust_repo=True,
)
get_speech_timestamps = vad_utils[0]
print("Silero VAD ready")


# =========================
# BASIC AUDIO OPS
# =========================
def highpass_filter(data, cutoff=80):
    nyq = 0.5 * config.SAMPLE_RATE
    b, a = butter(3, cutoff / nyq, btype="high")
    return lfilter(b, a, data)


def normalize_audio(audio, target_db=-20.0):
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    factor = 10.0 ** ((target_db - 20.0 * np.log10(rms)) / 20.0)
    return np.clip(audio * factor, -1.0, 1.0)


# =========================
# VAD
# =========================
def vad_prob(audio, sr=config.SAMPLE_RATE):
    if len(audio) == 0:
        return 0.0
    wav = torch.from_numpy(audio.astype("float32"))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return vad_model(wav, sr).item()


def extract_speech(audio, sr=config.SAMPLE_RATE):
    if len(audio) == 0:
        return audio

    wav = torch.from_numpy(audio.astype("float32"))
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        threshold=0.5,
    )

    if not timestamps:
        return np.array([], dtype=np.float32)

    segments = [audio[t["start"] : t["end"]] for t in timestamps]
    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


# =========================
# PIPELINES
# =========================
def process_realtime(x):
    """Realtime-safe pipeline"""
    x = highpass_filter(x)
    x = normalize_audio(x)
    return x


def process_final(x, sr=config.SAMPLE_RATE):
    """Offline final sentence cleanup"""
    x = process_realtime(x)
    x = nr.reduce_noise(
        y=x,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
    )
    return extract_speech(x, sr)


# =========================
# TEXT SAFETY (DROP ONLY)
# =========================
def filter_hallucination(
    text,
    duration,
    max_words_per_sec=3.0,
    min_unique_ratio=0.5,
):
    """
    Drop-only hallucination filter.
    Do NOT modify text here.
    """
    if not text:
        return ""

    words = text.split()
    if not words:
        return ""

    # too many words for duration
    if len(words) > int(duration * max_words_per_sec):
        return ""

    # repetition ratio
    if len(set(words)) / len(words) < min_unique_ratio:
        return ""

    # short audio but long text
    if duration < 0.5 and len(words) > 5:
        return ""

    # triple repeat
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return ""

    return text
