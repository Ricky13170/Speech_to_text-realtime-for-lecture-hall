import numpy as np
import torch
from silero_vad import load_silero_vad

SR = 16000
vad_model = load_silero_vad() 

def vad_prob_for_buffer(buf: np.ndarray):
    """
    buf: float32 or int16 chunk normalized to -1..1 or int16.
    returns a probability float in [0,1]
    """
    if buf.dtype != np.float32:
        buf = buf.astype(np.float32) / 32768.0
    # Silero expects 1D float tensor
    tensor = torch.from_numpy(buf).float()
    with torch.no_grad():
        out = vad_model(tensor, SR)
    try:
        prob = float(out.detach().cpu().numpy().squeeze())
    except Exception:
        prob = float(out)
    return prob