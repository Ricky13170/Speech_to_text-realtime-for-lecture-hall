# ASR Evaluation Test Scripts

Benchmarking scripts for streaming ASR models.

## Files

- `streaming_eval.py` - Main evaluation script (runs on Modal GPU)
- `generate_tables.py` - Generates summary tables from results

## Usage

### Run Evaluation

```bash
# Whisper + A100
modal run test/streaming_eval.py --dataset vlsp2020 --samples 100 --gpu a100 --model whisper

# PhoWhisper + H100
modal run test/streaming_eval.py --dataset vlsp2020 --samples 100 --gpu h100 --model phowhisper
```

### Options

| Argument | Values | Description |
|----------|--------|-------------|
| `--dataset` | `vlsp2020`, `earnings22` | Dataset to evaluate |
| `--samples` | int | Number of samples |
| `--gpu` | `a100`, `h100` | GPU type |
| `--model` | `whisper`, `phowhisper` | ASR model |

### Generate Tables

```bash
python test/generate_tables.py           # Markdown
python test/generate_tables.py --latex   # LaTeX
```

## Metrics

| Metric | Description |
|--------|-------------|
| WER | Word Error Rate (lower = better) |
| CER | Character Error Rate (lower = better) |
| TTFT | Time to First Token in ms |
| RTF | Real-Time Factor (<1.0 = faster than real-time) |

## Results

Results are saved to `test/results/{model}/{gpu}/{dataset}.json` and are gitignored.
