# 🎙️ Comparative Study on LSTM Networks for STT Conversion  
### 🧠 Research Project: Variations in Attention Mechanisms & Loss Functions

This repository implements an **LSTM-based Speech-to-Text (STT)** comparative study analyzing **attention mechanisms (Bahdanau, Multi-Head)** and **loss functions (CTC, Cross-Entropy)**.

---

## 🧠 Approach:
- **Baseline**: Encoder-decoder LSTM model.
- **Attention Variants**: Bahdanau, Multi-head (Transformer-inspired).
- **Loss Variants**: CTC loss vs Cross-Entropy.
- **Metrics**: Word Error Rate (WER), Character Error Rate (CER).

---

## 📂 Dataset:
- **LibriSpeech ASR Corpus (train-clean-100)**  
[Download Here](https://www.openslr.org/12/)

Organize as:
```
data/
 ├── train/
 │    ├── audio/
 │    └── transcripts/
 └── test/
      ├── audio/
      └── transcripts/
```

---

## 🚀 Getting Started:
1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/lstm-attention-stt-comparison.git
cd lstm-attention-stt-comparison
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Train models:
```bash
python src/train.py --model_type attention
```

4️⃣ Evaluate:
```bash
python src/evaluate.py
```

5️⃣ Inference:
```bash
python src/inference.py
```

---

## 📊 Highlights:
- Comparative study of **attention vs no-attention models**.
- Implements **CTC and Cross-Entropy loss functions**.
- Visualizable **attention weights** for interpretability.

---

## 📜 License:
MIT License © 2025
