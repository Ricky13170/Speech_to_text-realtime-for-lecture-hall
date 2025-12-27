# Real-time Vietnamese-English Speech Translation

Translate Vietnamese speech to English in real-time for lecture halls. Uses Whisper Large V3 for E2E transcription and translation.


## Requirements

- Windows 10/11
- Python 3.11+
- Modal account (free tier available)


## Setup with venv (Recommended)

```cmd
# Create virtual environment
python -m venv venv

# Activate venv
.\venv\Scripts\activate

# Install dependencies
pip install uv
uv pip install -r requirements.txt
uv sync

# Login to Modal 
modal setup

# Deploy to Modal
modal deploy server.py
python client.py
```


## Running the System

After deploying, you get a URL like:
```
https://YOUR_USERNAME--asr-thesis-asr-web.modal.run
```

Open this URL in your browser to use the app.


## Development

To make changes and redeploy:

```cmd
# Activate venv
.\venv\Scripts\activate

# Edit code, then deploy
modal deploy server.py
python client.py
```


## Usage

1. Select audio source: Microphone or Computer Audio
2. Click Record button
3. Speak in Vietnamese
4. View real-time transcription and translation
5. Click Stop when done


## Troubleshooting

No transcription output:
- Check microphone permission in browser
- Open F12 Console to see errors
- Speak louder and clearer

WebSocket connection error:
- Check Modal app status: modal app list
- Redeploy: modal deploy server.py

Cold start slow:
- First request takes 30-60s to load models
- Subsequent requests are fast


## Technical Specs

- ASR/Translation Model: openai/whisper-large-v3
- GPU: NVIDIA A10G (Modal Cloud)
- Sample Rate: 16kHz
- Window: 2s with 1s overlap
