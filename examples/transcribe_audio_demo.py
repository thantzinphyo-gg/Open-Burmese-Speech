"""
Quickstart: Burmese Speech Recognition (ASR) Inference
Author: Thant Zin Phyo
"""

import torch
from transformers import pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 1. Initialize Whisper-Small Burmese ASR Pipeline
# Other available models:
# - "thantzinphyo/Whisper-Small-ASR"
# - "thantzinphyo/Whisper-Base-ASR"
# - "thantzinphyo/Whisper-Tiny-ASR"
# - "thantzinphyo/Meta-MMS-300M-ASR"
# - "thantzinphyo/Wav2Vec2-XLS-R-300M-ASR"
model_id = "thantzinphyo/Whisper-Small-ASR"
print(f"Loading pipeline: {model_id}...")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=DEVICE,
)

def transcribe_file(audio_path: str):
    """Transcribes an input audio file (.wav, .mp3, etc.) to Burmese text."""
    result = asr_pipe(audio_path)
    print("\nTranscribed Burmese Text:")
    print("-" * 40)
    print(result["text"])
    print("-" * 40)
    return result["text"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1])
    else:
        print("Usage: python transcribe_audio_demo.py <path_to_audio.wav>")
