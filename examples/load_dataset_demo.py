"""
Quickstart: Loading Burmese Speech Datasets from Hugging Face
Author: Thant Zin Phyo
"""

from datasets import load_dataset

print("=" * 60)
print("1. Streaming Burmese Daily Dialogue Corpus (BDDC)")
print("=" * 60)

# Stream the BDDC dataset without downloading all 2.45 GB
dataset = load_dataset(
    "thantzinphyo/Burmese-Daily-Dialogue-Corpus",
    split="train",
    streaming=True
)

for idx, sample in enumerate(dataset.take(3), 1):
    print(f"\n[Sample {idx}]")
    print(f"Speaker : {sample.get('speaker', 'N/A')}")
    print(f"Text    : {sample.get('sentence', sample.get('text', 'N/A'))}")
    audio_info = sample.get("audio", {})
    print(f"Sampling Rate: {audio_info.get('sampling_rate')} Hz")
    print(f"Audio Array Shape: {len(audio_info.get('array', []))} samples")

print("\n" + "=" * 60)
print("2. Loading Burmese Speech Refined OpenSLR-80 (Streaming)")
print("=" * 60)

dataset_slr = load_dataset(
    "thantzinphyo/burmese-speech-refined-openslr-80",
    split="train",
    streaming=True
)

for idx, sample in enumerate(dataset_slr.take(2), 1):
    print(f"\n[OpenSLR-80 Sample {idx}]")
    print(f"Text: {sample.get('sentence', sample.get('text', 'N/A'))}")
