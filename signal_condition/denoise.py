import noisereduce as nr

def reduce_noise(x, sr):
    return nr.reduce_noise(
        y=x,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
    )
