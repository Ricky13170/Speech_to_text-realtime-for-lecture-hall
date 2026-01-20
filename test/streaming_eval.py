"""
Streaming ASR Evaluation Script

Evaluates ASR models (Whisper, PhoWhisper) on Vietnamese and English datasets
using streaming simulation with configurable buffer sizes.

Datasets:
- vlsp2020: Vietnamese (VLSP 2020 competition dataset)
- earnings22: English (Earnings calls with international accents)

Models:
- whisper: OpenAI Whisper large-v3
- phowhisper: VinAI PhoWhisper large (CT2 format)

Metrics (following SimulStreaming paper methodology):
- WER: Word Error Rate (lower is better)
- CER: Character Error Rate (lower is better)  
- TTFT: Time to First Token in ms (lower is better)
- AL: Average Lagging in ms (lower is better)
- RTF: Real-Time Factor (< 1.0 means faster than real-time)

Usage:
    modal run test/streaming_eval.py --dataset vlsp2020 --samples 100 --gpu a100 --model whisper
    modal run test/streaming_eval.py --dataset earnings22 --samples 100 --gpu h100 --model phowhisper

Results saved to: test/results/{model}/{gpu}/{dataset}.json
"""

import modal
import os
import re

app = modal.App("streaming-eval")

# Modal container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchcodec",
        "faster-whisper>=1.0.0",
        "ctranslate2>=4.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "jiwer>=3.0.0",
        "librosa>=0.10.0",
        "huggingface_hub>=0.20.0",
    )
    .add_local_dir("backend", remote_path="/root/backend")
)

# Dataset configurations
# Each dataset specifies HuggingFace path, split, column names, and language
DATASETS = {
    "vlsp2020": {
        "hf_name": "doof-ferb/vlsp2020_vinai_100h",
        "split": "train",  # Only train split available
        "audio_col": "audio",
        "text_col": "transcription",
        "language": "vi",
    },
    "earnings22": {
        "hf_name": "distil-whisper/earnings22",
        "config": "chunked",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "language": "en",
    },
}

# Model configurations
# Maps model name to HuggingFace model path for faster-whisper
MODELS = {
    "whisper": "large-v3",
    "phowhisper": "kiendt/PhoWhisper-large-ct2",
}

# Streaming simulation parameters
# Buffer size of 30s is optimal per Whisper paper (trained on 30s segments)
MAX_BUFFER_DURATION = 30.0  # seconds - accumulate audio until this duration
MIN_SILENCE_DURATION = 0.5   # seconds - minimum silence to trigger transcription


def normalize_text(text: str) -> str:
    """
    Normalize text for fair WER/CER comparison.
    - Convert to lowercase
    - Remove punctuation (preserving Vietnamese diacritics)
    - Normalize whitespace
    """
    if not text:
        return ""
    text = text.lower()
    # Keep Vietnamese characters with diacritics
    text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', text)
    text = ' '.join(text.split())
    return text.strip()


def run_evaluation(dataset_name: str, max_samples: int, model: str):
    """
    Core evaluation logic - runs on Modal container.
    
    Steps:
    1. Load ASR model (Whisper or PhoWhisper)
    2. Load dataset samples from HuggingFace
    3. Run streaming simulation
    4. Calculate metrics
    """
    import sys
    sys.path.insert(0, "/root")
    
    import time
    import numpy as np
    from huggingface_hub import login
    from faster_whisper import WhisperModel
    import os
    
    # Authenticate with HuggingFace for private datasets
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    from jiwer import wer, cer
    from backend.vad import EnergyVAD
    from backend.config import VAD_THRESHOLD
    
    # Validate inputs
    if dataset_name not in DATASETS:
        return {"error": f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"}
    
    if model not in MODELS:
        return {"error": f"Unknown model: {model}. Available: {list(MODELS.keys())}"}
    
    config = DATASETS[dataset_name]
    model_path = MODELS[model]
    
    print(f"\n{'='*60}")
    print(f"STREAMING ASR EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model} ({model_path})")
    print(f"Dataset: {config['hf_name']}")
    print(f"Language: {config['language']}")
    print(f"Samples: {max_samples}")
    print(f"Buffer: {MAX_BUFFER_DURATION}s")
    
    # Step 1: Load ASR model
    print("\n[1/3] Loading ASR model...")
    asr_model = WhisperModel(
        model_path,
        device="cuda",
        compute_type="float16",  # Use FP16 for faster inference on GPU
    )
    vad = EnergyVAD(threshold=VAD_THRESHOLD)
    print("  Done")
    
    # Step 2: Load dataset samples
    print("\n[2/3] Loading dataset...")
    samples = load_dataset_samples(config, max_samples)
    print(f"  Loaded {len(samples)} samples")
    
    if not samples:
        return {"error": "No samples loaded", "dataset": dataset_name}
    
    # Step 3: Run streaming simulation
    print("\n[3/3] Running streaming simulation...")
    results = run_streaming_simulation(samples, asr_model, vad, config["language"])
    
    # Calculate WER/CER
    if results["references"] and results["hypotheses"]:
        results["wer"] = wer(results["references"], results["hypotheses"])
        results["cer"] = cer(results["references"], results["hypotheses"])
    
    # Aggregate statistics
    results["model"] = model
    results["dataset"] = dataset_name
    results["language"] = config["language"]
    results["num_samples"] = len(results["references"])
    results["avg_ttft_ms"] = np.mean(results["ttft"]) * 1000 if results["ttft"] else 0
    results["avg_al_ms"] = np.mean(results["al"]) * 1000 if results["al"] else 0
    results["avg_rtf"] = np.mean(results["rtf"]) if results["rtf"] else 0
    results["avg_latency_ms"] = np.mean(results["latency"]) * 1000 if results["latency"] else 0
    results["p90_latency_ms"] = np.percentile(results["latency"], 90) * 1000 if results["latency"] else 0
    
    # Print results summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model} on {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {results['num_samples']}")
    print(f"WER: {results.get('wer', 0)*100:.2f}%")
    print(f"CER: {results.get('cer', 0)*100:.2f}%")
    print(f"TTFT: {results['avg_ttft_ms']:.1f}ms")
    print(f"AL: {results['avg_al_ms']:.1f}ms")
    print(f"RTF: {results['avg_rtf']:.3f}x")
    
    # Return results without raw data (too large)
    return {k: v for k, v in results.items() 
            if k not in ["references", "hypotheses", "ttft", "al", "rtf", "latency"]}


# GPU-specific Modal functions
# These are separate functions to allow Modal to allocate the correct GPU type

@app.function(gpu="A100", timeout=3600, image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate_a100(dataset_name: str, max_samples: int = 100, model: str = "whisper"):
    """Run evaluation on NVIDIA A100 GPU"""
    return run_evaluation(dataset_name, max_samples, model)


@app.function(gpu="H100", timeout=3600, image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate_h100(dataset_name: str, max_samples: int = 100, model: str = "whisper"):
    """Run evaluation on NVIDIA H100 GPU"""
    return run_evaluation(dataset_name, max_samples, model)


def load_dataset_samples(config: dict, max_samples: int):
    """
    Load audio samples from HuggingFace dataset.
    
    Uses streaming mode to avoid downloading entire dataset.
    Resamples audio to 16kHz if needed.
    """
    from datasets import load_dataset
    import numpy as np
    import librosa
    
    samples = []
    
    # Build load arguments
    load_kwargs = {"split": config["split"], "streaming": True}
    if config.get("config"):
        load_kwargs["name"] = config["config"]
    
    # Load dataset in streaming mode
    ds = load_dataset(config["hf_name"], **load_kwargs)
    
    # Fetch items
    items = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        items.append(item)
    
    audio_col = config["audio_col"]
    text_col = config["text_col"]
    
    # Process each item
    for i, item in enumerate(items):
        try:
            # Get transcription text
            text = str(item.get(text_col, "")).strip()
            if not text or len(text) < 3:
                continue
            
            # Decode audio
            audio_data = item[audio_col]
            audio, sr = decode_audio(audio_data)
            
            if audio is None or len(audio) < 8000:  # Skip very short audio
                continue
            
            # Resample to 16kHz if needed
            if sr != 16000 and sr > 0:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            samples.append({"audio": audio, "text": text})
        except Exception as e:
            if i == 0:
                print(f"  Warning: Failed to process item 0: {e}")
            continue
    
    return samples


def decode_audio(audio_data):
    """
    Decode audio from various HuggingFace dataset formats.
    
    Handles:
    - AudioDecoder (torchcodec): Use get_all_samples()
    - Dictionary with 'array': Direct numpy array
    - Dictionary with 'bytes': Decode from bytes
    - Tensor-like objects: Convert to numpy
    """
    import numpy as np
    
    audio = None
    sr = 16000
    
    # Format 1: AudioDecoder from torchcodec
    if hasattr(audio_data, 'get_all_samples'):
        samples_obj = audio_data.get_all_samples()
        if hasattr(samples_obj, 'data'):
            data = samples_obj.data
            if hasattr(data, 'numpy'):
                audio = data.numpy().flatten().astype(np.float32)
            else:
                audio = np.asarray(data, dtype=np.float32).flatten()
            sr = getattr(samples_obj, 'sample_rate', 16000)
    
    # Format 2: Dictionary format
    elif isinstance(audio_data, dict):
        if "array" in audio_data:
            arr = audio_data["array"]
            sr = audio_data.get("sampling_rate", 16000)
            if hasattr(arr, 'numpy'):
                audio = arr.numpy().flatten().astype(np.float32)
            else:
                audio = np.asarray(arr, dtype=np.float32).flatten()
        elif "bytes" in audio_data:
            import soundfile as sf
            import io
            audio, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            audio = audio.astype(np.float32).flatten()
    
    # Format 3: Tensor-like
    elif hasattr(audio_data, 'numpy'):
        audio = audio_data.numpy().flatten().astype(np.float32)
    
    return audio, sr


def run_streaming_simulation(samples, asr_model, vad, language):
    """
    Simulate streaming ASR processing.
    
    Processes audio in 100ms chunks, accumulating into a buffer.
    Triggers transcription when:
    - Buffer exceeds MAX_BUFFER_DURATION, or
    - Silence detected after speech > MIN_SILENCE_DURATION
    """
    import time
    import numpy as np
    
    chunk_samples = int(16000 * 100 / 1000)  # 100ms = 1600 samples at 16kHz
    
    results = {
        "references": [],
        "hypotheses": [],
        "ttft": [],
        "al": [],
        "rtf": [],
        "latency": [],
        "wer": 0.0,
        "cer": 0.0,
    }
    
    for idx, sample in enumerate(samples):
        audio = sample["audio"]
        reference = sample["text"]
        
        # Simulate streaming for this sample
        session_result = simulate_session(audio, asr_model, vad, language, chunk_samples)
        
        if session_result["text"]:
            # Normalize both reference and hypothesis for fair comparison
            ref_norm = normalize_text(reference)
            hyp_norm = normalize_text(session_result["text"])
            
            if ref_norm and hyp_norm:
                results["references"].append(ref_norm)
                results["hypotheses"].append(hyp_norm)
                results["ttft"].append(session_result["ttft"])
                results["al"].append(session_result["al"])
                results["rtf"].append(session_result["rtf"])
                results["latency"].append(session_result["latency"])
        
        # Progress logging
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(samples)}")
    
    return results


def simulate_session(audio, asr_model, vad, language, chunk_samples):
    """
    Simulate a single streaming session.
    
    Returns:
    - text: Final transcription
    - ttft: Time to first token (seconds)
    - al: Average lagging (seconds) - how far behind real-time
    - rtf: Real-time factor - processing_time / audio_duration
    - latency: Total processing time (seconds)
    """
    import time
    import numpy as np
    
    audio_buffer = np.array([], dtype=np.float32)
    audio_duration = len(audio) / 16000
    num_chunks = max(1, len(audio) // chunk_samples)
    chunk_duration = chunk_samples / 16000  # 0.1 seconds
    
    session_start = time.time()
    first_emission_time = None
    emissions = []  # List of (wall_clock_time, simulated_audio_time)
    silence_counter = 0.0
    simulated_time = 0.0
    final_text = ""
    
    # Process audio chunk by chunk
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start_idx:end_idx]
        simulated_time = (i + 1) * chunk_duration
        
        # Voice Activity Detection
        is_speech = vad.is_speech(chunk)
        
        if is_speech:
            silence_counter = 0.0
            audio_buffer = np.concatenate([audio_buffer, chunk])
        else:
            silence_counter += chunk_duration
            # Include some silence in buffer for context
            if silence_counter < MIN_SILENCE_DURATION:
                audio_buffer = np.concatenate([audio_buffer, chunk])
        
        buffer_dur = len(audio_buffer) / 16000
        
        # Decide when to transcribe
        should_finalize = (
            (silence_counter > MIN_SILENCE_DURATION and buffer_dur > 0.5) or
            buffer_dur > MAX_BUFFER_DURATION
        )
        
        if should_finalize and len(audio_buffer) > 4000:
            emit_wall_time = time.time() - session_start
            
            # Transcribe buffered audio
            segments, _ = asr_model.transcribe(
                audio_buffer,
                language=language,
                beam_size=5,
                vad_filter=True,
                without_timestamps=True,
            )
            text = " ".join([s.text.strip() for s in segments]).strip()
            
            if text:
                emissions.append((emit_wall_time, simulated_time))
                if first_emission_time is None:
                    first_emission_time = emit_wall_time
                final_text = text
            
            audio_buffer = np.array([], dtype=np.float32)
    
    # Process remaining audio
    if len(audio_buffer) > 4000:
        emit_wall_time = time.time() - session_start
        segments, _ = asr_model.transcribe(
            audio_buffer,
            language=language,
            beam_size=5,
            vad_filter=True,
            without_timestamps=True,
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        if text:
            emissions.append((emit_wall_time, audio_duration))
            if first_emission_time is None:
                first_emission_time = emit_wall_time
            final_text = text
    
    total_time = time.time() - session_start
    
    # Calculate metrics
    ttft = first_emission_time if first_emission_time else audio_duration
    al = np.mean([w - s for w, s in emissions]) if emissions else total_time
    rtf = total_time / audio_duration if audio_duration > 0 else 1.0
    
    return {"text": final_text, "ttft": ttft, "al": al, "rtf": rtf, "latency": total_time}


@app.local_entrypoint()
def main(dataset: str = "vlsp2020", samples: int = 100, gpu: str = "a100", model: str = "whisper"):
    """
    Main entry point for streaming evaluation.
    
    Args:
        dataset: Dataset name (vlsp2020, earnings22)
        samples: Number of samples to evaluate
        gpu: GPU type (a100, h100)
        model: Model name (whisper, phowhisper)
    """
    import json
    from pathlib import Path
    
    print(f"\n{'#'*60}")
    print(f"# Model: {model.upper()} | GPU: {gpu.upper()} | Dataset: {dataset}")
    print(f"{'#'*60}")
    
    # Call appropriate GPU function
    if gpu.lower() == "a100":
        result = evaluate_a100.remote(dataset, samples, model)
    else:
        result = evaluate_h100.remote(dataset, samples, model)
    
    # Save results to JSON
    results_dir = Path(__file__).parent / "results" / model.lower() / gpu.lower()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"{dataset}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved: {output_file}")
    
    # Print summary
    if "error" not in result:
        print(f"\n{'='*60}")
        print(f"Model: {result['model']} | Dataset: {result['dataset']}")
        print(f"WER: {result.get('wer',0)*100:.2f}% | CER: {result.get('cer',0)*100:.2f}%")
        print(f"TTFT: {result['avg_ttft_ms']:.1f}ms | RTF: {result['avg_rtf']:.3f}x")
        print(f"{'='*60}")
