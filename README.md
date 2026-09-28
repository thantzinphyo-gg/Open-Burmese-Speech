# Open-Burmese-Speech
**Open Corpora and Acoustic Models for the Myanmar Language**

*Author: [Thant Zin Phyo](https://huggingface.co/thantzinphyo)*

[![Hugging Face Datasets](https://img.shields.io/badge/Hugging_Face-BDDC_Dataset-yellow.svg)](https://huggingface.co/datasets/thantzinphyo/Burmese-Daily-Dialogue-Corpus)
[![Hugging Face Profile](https://img.shields.io/badge/Hugging_Face-thantzinphyo-blue.svg)](https://huggingface.co/thantzinphyo)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)

---

## Overview

**Open-Burmese-Speech** is an open-source speech processing ecosystem designed to advance Automatic Speech Recognition (ASR), Speech Synthesis (TTS), and audio language processing for Myanmar (Burmese).

Audio datasets are permanently hosted on the **Hugging Face Hub** with streaming capabilities, while this repository maintains documentation, usage examples, and model references.

**Open-Burmese-Speech** သည် မြန်မာဘာသာစကားအတွက် အလိုအလျောက် စကားပြောအသံ အသိအမှတ်ပြုစနစ် (ASR) နှင့် အသံဆိုင်ရာ နည်းပညာ သုတေသနများအတွက် ရည်ရွယ်တည်ဆောက်ထားသော Open-Source Corpus နှင့် Model Hub ဖြစ်ပါသည်။

---

## Dataset Collection

| Dataset Name | Domain / Type | Duration | Speakers | Audio Format | Hugging Face Hub |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Burmese Daily Dialogue Corpus (BDDC)** | Conversational / Daily Dialogue | **~22.38 Hours** (24,560 WAVs) | 13 Synthetic Speakers | 16 kHz Mono PCM | [View on Hugging Face](https://huggingface.co/datasets/thantzinphyo/Burmese-Daily-Dialogue-Corpus) |
| **Burmese Speech Refined OpenSLR-80** | Crowdsourced General Speech | **~2.47 Hours** (Refined Subset) | Single-speaker | 16 kHz Mono PCM | [View on Hugging Face](https://huggingface.co/datasets/thantzinphyo/burmese-speech-refined-openslr-80) |

---

## Model Zoo

Acoustic speech recognition models fine-tuned for Myanmar language:

| Model Name | Architecture / Base Model | Hugging Face Hub Link |
| :--- | :--- | :---: |
| **Dolphin-Base-Burmese-ASR** | E-Branchformer + Transformer (`DataoceanAI/dolphin-base`) | [Hugging Face](https://huggingface.co/thantzinphyo/Dolphin-Base-Burmese-ASR) |
| **Meta-MMS-300M-ASR** | Facebook MMS-300M (Wav2Vec2) | [Hugging Face](https://huggingface.co/thantzinphyo/Meta-MMS-300M-ASR) |
| **Whisper-Small-ASR** | OpenAI Whisper Small (244M) | [Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Small-ASR) |
| **Wav2Vec2-XLS-R-300M-ASR** | Facebook XLS-R 300M | [Hugging Face](https://huggingface.co/thantzinphyo/Wav2Vec2-XLS-R-300M-ASR) |
| **Whisper-Base-ASR** | OpenAI Whisper Base (74M) | [Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Base-ASR) |
| **Whisper-Tiny-ASR** | OpenAI Whisper Tiny (39M) | [Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Tiny-ASR) |

---

## Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/thantzinphyo-gg/Open-Burmese-Speech.git
cd Open-Burmese-Speech
pip install -r requirements.txt
```

### 2. Stream Audio & Transcripts from Hugging Face

You can stream samples directly from the Hugging Face Hub without downloading the entire dataset:

```python
from datasets import load_dataset

# Stream Burmese Daily Dialogue Corpus (BDDC)
dataset = load_dataset(
    "thantzinphyo/Burmese-Daily-Dialogue-Corpus",
    split="train",
    streaming=True
)

for sample in dataset.take(3):
    print("Speaker:", sample["speaker"])
    print("Burmese Text:", sample["sentence"])
    print("Audio Sampling Rate:", sample["audio"]["sampling_rate"])
```

### 3. Speech Recognition Inference

Run Burmese ASR inference via Hugging Face Transformers pipeline:

```python
import torch
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="thantzinphyo/Whisper-Small-ASR",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

result = pipe("your_burmese_speech.wav")
print("Transcribed Text:", result["text"])
```

---

## Myanmar Language Summary

### ဒေတာအတွဲဆိုင်ရာ အချက်အလက်များ
ဤ Repository ပါ Dataset များနှင့် Acoustic Model များသည် မြန်မာဘာသာစကား သုတေသနနှင့် အသံဆိုင်ရာ Artificial Intelligence လုပ်ငန်းများအတွက် ရည်ရွယ်ထုတ်ဝေထားခြင်း ဖြစ်ပါသည်။

- **လိုင်စင် (License):** Creative Commons Attribution 4.0 International (CC BY 4.0) လိုင်စင်ဖြင့် ဖြန့်ချိထားပြီး သုတေသန၊ ပညာရေးနှင့် စီးပွားဖြစ် Model Training များအတွက် လွတ်လပ်စွာ အခမဲ့ အသုံးပြုနိုင်ပါသည်။
- **အသိအမှတ်ပြုခြင်း (Attribution):** သုတေသနလုပ်ငန်းများတွင် ထည့်သွင်းအသုံးပြုပါက သင့်လျော်သော Citation/Credit ပေးရန် လိုအပ်ပါသည်။

---

## Citation

If you use these corpora or models in your research or applications, please cite as follows:

```bibtex
@misc{open_burmese_speech_2026,
  author       = {Thant Zin Phyo},
  title        = {Open-Burmese-Speech: Open Corpora and Acoustic Models for Myanmar Language},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/thantzinphyo-gg/Open-Burmese-Speech}}
}

@misc{bddc_corpus_2026,
  author       = {Thant Zin Phyo},
  title        = {Burmese Daily Dialogue Corpus (BDDC)},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/thantzinphyo/Burmese-Daily-Dialogue-Corpus}}
}
```

---

## Author & Contact

* **Author:** Thant Zin Phyo
* **Hugging Face:** [@thantzinphyo](https://huggingface.co/thantzinphyo)
* **GitHub:** [@thantzinphyo-gg](https://github.com/thantzinphyo-gg)
