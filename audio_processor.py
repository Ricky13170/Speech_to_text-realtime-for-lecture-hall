import torch
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import config

print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
get_speech_timestamps = vad_utils[0]
print("Silero VAD ready")




def highpass_filter(data, cutoff=80):
    nyq = 0.5 * config.SAMPLE_RATE
    b, a = butter(3, cutoff / nyq, btype='high')
    return lfilter(b, a, data)




def normalize_audio(audio, target_db=-20.0):
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    factor = 10.0 ** ((target_db - 20.0 * np.log10(rms)) / 20.0)
    return np.clip(audio * factor, -1.0, 1.0)



def vad_prob(audio, sr=config.SAMPLE_RATE):
    if len(audio) == 0:
        return 0.0
    wav = torch.from_numpy(audio.astype('float32'))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return vad_model(wav, sr).item()




def extract_speech(audio, sr=config.SAMPLE_RATE):
    if len(audio) == 0:
        return audio
    wav = torch.from_numpy(audio.astype('float32'))
    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr, threshold=0.5)
    if not timestamps:
        return np.array([], dtype=np.float32)
    segments = [audio[ts['start']:ts['end']] for ts in timestamps]
    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)



def process_realtime(x):
    return normalize_audio(highpass_filter(x))



def process_final(x, sr=config.SAMPLE_RATE):
    x = process_realtime(x)
    x = nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=0.75)
    return extract_speech(x, sr)




def local_agreement(history, min_agree=2):
    if len(history) < min_agree:
        return ""
    recent = [s.split() for s in history[-min_agree:] if s]
    if not recent or not all(recent):
        return ""
    agreed = []
    for i in range(min(len(w) for w in recent)):
        if all(w[i] == recent[0][i] for w in recent):
            agreed.append(recent[0][i])
        else:
            break
    return " ".join(agreed)




def filter_hallucination(text, duration, max_words_per_sec=5.0, min_unique_ratio=0.3):
    if not text:
        return ""
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return ""
    
    expected_max = int(duration * max_words_per_sec)
    if word_count > expected_max:
        return ""
    
    unique_ratio = len(set(words)) / word_count
    if unique_ratio < min_unique_ratio:
        return ""
    
    if duration < 0.5 and word_count > 5:
        return ""
    
    return text
