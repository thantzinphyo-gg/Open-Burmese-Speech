# Open-Burmese-Speech 🇲🇲🎙️
### Comprehensive Open-Source Burmese Speech Corpora, Benchmarks & Fine-Tuned ASR Models
**Curated & Maintained by [Thant Zin Phyo](https://huggingface.co/thantzinphyo)**

[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-BDDC%20Dataset-yellow.svg)](https://huggingface.co/datasets/thantzinphyo/Burmese-Daily-Dialogue-Corpus)
[![Hugging Face Profile](https://img.shields.io/badge/🤗%20Hugging%20Face-thantzinphyo-blue.svg)](https://huggingface.co/thantzinphyo)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Myanmar Unicode Standard](https://img.shields.io/badge/Unicode-Myanmar%20Standard-orange.svg)](#word-segmentation--text-normalization)

---

## 📌 Overview / စီမံကိန်း မိတ်ဆက်

**Open-Burmese-Speech** is a unified open-source ecosystem designed to advance speech recognition (ASR), speech synthesis (TTS), and audio language processing for the Myanmar (Burmese) language.

All raw audio files and large parquet shards are permanently hosted on **Hugging Face Hub** with free, high-speed streaming support, while this repository serves as the official open-source documentation, developer specification, and evaluation benchmark hub.

**Open-Burmese-Speech** သည် မြန်မာဘာသာစကားအတွက် Automatic Speech Recognition (ASR)၊ စကားပြောအသံ အသိအမှတ်ပြုစနစ်နှင့် သဘာဝဘာသာစကား သုတေသနများအတွက် ရည်ရွယ်ဖန်တီးထားသော Open-Source Corpus နှင့် Model Hub ဖြစ်ပါသည်။ 

---

## 📊 Dataset Collection / မြန်မာစကားပြော ဒေတာအတွဲများ

| Dataset Name | Domain / Type | Duration | Speakers | Audio Format | Hugging Face Hub Link |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Burmese Daily Dialogue Corpus (BDDC)** | Conversational / Daily Dialogue | **~22.38 Hours** (24,560 WAVs) | 13 Synthetic Speakers | 16 kHz Mono PCM | [🤗 View on Hugging Face](https://huggingface.co/datasets/thantzinphyo/Burmese-Daily-Dialogue-Corpus) |
| **Burmese Speech Refined OpenSLR-80** | Crowdsourced General Speech | **~30 Hours** (Standardized) | Multi-speaker | 16 kHz Mono PCM | [🤗 View on Hugging Face](https://huggingface.co/datasets/thantzinphyo/burmese-speech-refined-openslr-80) |
| **Myanmar ShopVoice** | Retail & E-Commerce Conversational | Domain Specific | Multi-speaker | 16 kHz Mono PCM | [🤗 View on Hugging Face](https://huggingface.co/datasets/thantzinphyo/myanmar-shopvoice) |

---

## 🏆 Model Zoo & ASR Benchmark Leaderboard

All models below were fine-tuned using standardized Myanmar Unicode text preprocessing on the **Burmese Daily Dialogue Corpus (BDDC)**. Evaluation was conducted across both the Validation split and the **Held-Out Test Set (Unseen Zero-Shot Speakers: ဂီတ & နန္ဒ)**.

| Model Name | Architecture / Base | Val CER (%) | Val WER (%) | Val chrF | Test CER (%) *(Unseen)* | Test WER (%) *(Unseen)* | Test chrF *(Unseen)* | Model Link |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dolphin-Base-Burmese-ASR** | E-Branchformer + Transformer | **3.00** | 22.88 | **95.54** | **4.42** | **23.51** | **92.54** | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Dolphin-Base-Burmese-ASR) |
| **Meta-MMS-300M-ASR** | Facebook MMS-300M (Wav2Vec2) | **2.85** | **20.66** | 95.39 | 6.66 | 32.64 | 87.22 | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Meta-MMS-300M-ASR) |
| **Whisper-Small-ASR** | OpenAI Whisper Small (244M) | 3.21 | 20.70 | 94.86 | 5.74 | 28.22 | 89.93 | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Small-ASR) |
| **Wav2Vec2-XLS-R-300M-ASR** | Facebook XLS-R 300M | 3.53 | 24.25 | 94.97 | 6.58 | 33.24 | 87.69 | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Wav2Vec2-XLS-R-300M-ASR) |
| **Whisper-Base-ASR** | OpenAI Whisper Base (74M) | 4.43 | 26.55 | 91.94 | 8.55 | 36.29 | 83.71 | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Base-ASR) |
| **Whisper-Tiny-ASR** | OpenAI Whisper Tiny (39M) | 4.47 | 25.21 | 92.91 | 11.69 | 42.29 | 79.82 | [🤗 Hugging Face](https://huggingface.co/thantzinphyo/Whisper-Tiny-ASR) |

*Older generation models fine-tuned on OpenSLR-80:*
- [Whisper-Small-Myanmar-Partial-Freezing](https://huggingface.co/thantzinphyo/Whisper-Small-Myanmar-Partial-Freezing) *(1,300+ downloads)*
- [Whisper-Tiny-Myanmar-Full-Fine-Tune](https://huggingface.co/thantzinphyo/Whisper-Tiny-Myanmar-Full-Fine-Tune)

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/thantzinphyo-gg/Open-Burmese-Speech.git
cd Open-Burmese-Speech
pip install -r requirements.txt
```

### 2. Stream Audio & Transcripts from Hugging Face

You do not need to download the full 2.45 GB dataset to begin experimenting. Stream it directly:

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

### 3. Speech Recognition (Transcribing Burmese Audio)

Transcribe any Burmese audio file in 2 lines with Transformers:

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

## 📐 Word Segmentation & Text Normalization

A critical bottleneck in Burmese Speech Recognition is word boundary ambiguity. All datasets in this repository follow unified word segmentation and standardization rules:
1. **Unicode Strictness:** Standard Myanmar Unicode (`U+1000` to `U+109F`), strictly rejecting legacy pseudo-Zawgyi encodings.
2. **Compound Word Grouping:** Cohesive grammatical and lexical words are segmented with whitespace delimiters to ensure optimal Byte-Pair Encoding (BPE) tokenization.
3. **Punctuation Cleanliness:** Extraneous quotation marks and non-standard symbols are removed to ensure zero acoustic mismatch.

For the complete technical specification, please refer to [WORD_SEGMENTATION_RULES.md](WORD_SEGMENTATION_RULES.md).

---

## 🇲🇲 မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်

### Dataset အကြောင်း
ဤ Repository ရှိ Dataset များနှင့် Fine-tuned Model များသည် မြန်မာဘာသာစကား သုတေသနနှင့် Voice AI Applications များ (Call Center Automation, Voice Assistant, Transcriber) ဖန်တီးရာတွင် အထောက်အကူပြုနိုင်ရန် ဖန်တီးထားခြင်း ဖြစ်ပါသည်။

- **အသုံးပြုခွင့် (License):** Creative Commons Attribution 4.0 International (CC BY 4.0) လိုင်စင်ဖြင့် ထုတ်ဝေထားသောကြောင့် ပညာရေး၊ သုတေသနနှင့် Commercial Model Training များအတွက် လွတ်လပ်စွာ အခမဲ့ အသုံးပြုနိုင်ပါသည်။
- **အသိအမှတ်ပြုခြင်း (Attribution):** မည်သည့် ပရောဂျက်တွင်မဆို ထည့်သွင်းအသုံးပြုပါက အောက်ဖော်ပြပါ Citation အတိုင်း Credit ပေးနိုင်ပါသည်။

---

## 📖 Citation

If you use these datasets or models in your academic research or commercial products, please cite as follows:

```bibtex
@misc{open_burmese_speech_2026,
  author       = {Thant Zin Phyo},
  title        = {Open-Burmese-Speech: Open Corpora, Benchmarks and Acoustic Models for Myanmar Language},
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

## 👤 Author & Acknowledgements

* **Author:** Thant Zin Phyo
* **Hugging Face Hub:** [@thantzinphyo](https://huggingface.co/thantzinphyo)
* **GitHub:** [@thantzinphyo-gg](https://github.com/thantzinphyo-gg)
