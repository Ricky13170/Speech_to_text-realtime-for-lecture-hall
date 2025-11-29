import torch
import noisereduce as nr
import numpy as np
from scipy.signal import butter, lfilter
import config

# --- 0. UTILS ---
def highpass_filter(data, cutoff=80, fs=config.SAMPLE_RATE, order=3):
    # Loc bo tan so thap (tieng on nen nhu quat, gio)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

def normalize_audio(audio, target_dbfs=-20.0):
    # Chuan hoa am luong ve muc tieu chuan de AI de nghe
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    rms = (audio**2).mean() ** 0.5
    if rms < 1e-6:
        return audio
    current_db = 20.0 * np.log10(rms + 1e-12)
    change_db = target_dbfs - current_db
    factor = 10.0 ** (change_db / 20.0)
    out = audio * factor
    out = np.clip(out, -1.0, 1.0)
    return out

# --- 1. VAD SETUP ---
print("Loading Silero VAD...")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("Silero VAD ready")

def vad_prob_for_buffer(audio_float32, sr=config.SAMPLE_RATE):
    # Tinh xac suat co giong noi (0.0 - 1.0)
    if len(audio_float32) == 0:
        return 0.0
    
    if audio_float32.dtype != 'float32':
         audio_float32 = audio_float32.astype('float32')

    wav = torch.from_numpy(audio_float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
        
    speech_prob = model_vad(wav, sr).item()
    return speech_prob

# --- 2. AUDIO PROCESSING ---

def process_realtime_chunk(x):
    # Xu ly nhanh (realtime): chi loc va chuan hoa
    x = highpass_filter(x)
    x = normalize_audio(x)
    return x

def process_final_sentence(x, sr=config.SAMPLE_RATE):
    # Xu ly ky (final): them giam nhieu sau (noise reduction)
    x = process_realtime_chunk(x)
    x = nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=0.75)
    return x
