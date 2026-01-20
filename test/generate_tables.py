"""
Generate Summary Tables from Evaluation Results

Reads JSON results and generates a single combined table.

Usage:
    python test/generate_tables.py           # Markdown
    python test/generate_tables.py --latex   # LaTeX
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = Path(__file__).parent / "tables"


def load_results():
    """Load all result files."""
    results = {}
    
    for model in ["whisper", "phowhisper"]:
        results[model] = {"a100": {}, "h100": {}}
        for gpu in ["a100", "h100"]:
            gpu_dir = RESULTS_DIR / model / gpu
            if gpu_dir.exists():
                for f in gpu_dir.glob("*.json"):
                    with open(f) as file:
                        data = json.load(file)
                        if "error" not in data:
                            results[model][gpu][f.stem] = data
    
    return results


def generate_markdown(results: dict):
    """Generate single combined Markdown table."""
    lines = []
    lines.append("# Streaming ASR Evaluation Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Combined Results")
    lines.append("")
    lines.append("| Model | Dataset | GPU | WER | CER | TTFT | RTF |")
    lines.append("|-------|---------|-----|-----|-----|------|-----|")
    
    for model_name, model_label in [("whisper", "Whisper"), ("phowhisper", "PhoWhisper")]:
        for gpu in ["a100", "h100"]:
            for ds, data in sorted(results[model_name][gpu].items()):
                wer = f"{data.get('wer', 0)*100:.2f}%"
                cer = f"{data.get('cer', 0)*100:.2f}%"
                ttft = f"{data.get('avg_ttft_ms', 0):.0f}ms"
                rtf = f"{data.get('avg_rtf', 0):.3f}x"
                lines.append(f"| {model_label} | {ds} | {gpu.upper()} | {wer} | {cer} | {ttft} | {rtf} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Metrics:**")
    lines.append("- WER: Word Error Rate (lower = better)")
    lines.append("- CER: Character Error Rate (lower = better)")
    lines.append("- TTFT: Time to First Token (lower = better)")
    lines.append("- RTF: Real-Time Factor (<1.0 = faster than real-time)")
    
    return "\n".join(lines)


def generate_latex(results: dict):
    """Generate LaTeX table."""
    lines = []
    lines.append("% Streaming ASR Evaluation Results")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Streaming ASR Evaluation Results}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\hline")
    lines.append("Model & Dataset & GPU & WER & CER & TTFT & RTF \\\\")
    lines.append("\\hline")
    
    for model_name, model_label in [("whisper", "Whisper"), ("phowhisper", "PhoWhisper")]:
        for gpu in ["a100", "h100"]:
            for ds, data in sorted(results[model_name][gpu].items()):
                wer = f"{data.get('wer', 0)*100:.2f}\\%"
                cer = f"{data.get('cer', 0)*100:.2f}\\%"
                ttft = f"{data.get('avg_ttft_ms', 0):.0f}ms"
                rtf = f"{data.get('avg_rtf', 0):.3f}x"
                lines.append(f"{model_label} & {ds} & {gpu.upper()} & {wer} & {cer} & {ttft} & {rtf} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    import sys
    
    format = "latex" if "--latex" in sys.argv else "markdown"
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    
    if format == "latex":
        content = generate_latex(results)
        ext = "tex"
    else:
        content = generate_markdown(results)
        ext = "md"
    
    print(content)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TABLES_DIR / f"results_{timestamp}.{ext}"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
