# 🧠 RNN-Based General Knowledge Q&A Model

A simple Recurrent Neural Network (RNN) built from scratch using **PyTorch** that learns to answer general knowledge questions — trained entirely on a custom GK dataset covering geography, science, history, tech, and more.

---

## 📌 Project Overview

This project was built while learning the fundamentals of RNNs. The goal was to understand how sequential models process text by building a complete pipeline: from raw text to a trained model that can answer factual questions.

**Input:** A natural language question (e.g., *"Which planet is known as the red planet?"*)  
**Output:** A predicted answer token (e.g., *"mars"*)

---

## 🗂️ Dataset

**File:** `gk_qna_dataset.csv`

| Property | Value |
|---|---|
| Total samples | 171 Q&A pairs |
| Unique answers | 72 classes |
| Vocabulary size | 243 tokens |
| Categories | Capitals, Science, History, Literature, Tech, Animals, Sports |

Sample entries:
```
What is the capital of France?     → Paris
Which planet is the red planet?    → Mars
Who discovered gravity?            → Isaac Newton
What is H2O commonly called?       → Water
Which bird cannot fly?             → Ostrich
```

---

## 🏗️ Architecture

```
Input Question
      ↓
  Tokenizer  (lowercase → remove punctuation → split)
      ↓
 Embedding Layer  (vocab_size=243, embedding_dim=50)
      ↓
   RNN Layer  (hidden_size=64, batch_first=True)
      ↓
  Final Hidden State
      ↓
  Linear (FC) Layer  (hidden_size=64 → vocab_size=243)
      ↓
 Predicted Answer Token
```

### Model Parameters

| Component | Config |
|---|---|
| Embedding Dimension | 50 |
| RNN Hidden Size | 64 |
| Output Classes | 243 (full vocab) |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 50 |
| Batch Size | 1 |

---

## 📉 Training Progress

| Epoch | Loss |
|---|---|
| 10 | 7.6447 |
| 20 | 1.6165 |
| 30 | 0.5431 |
| 40 | 0.2093 |
| 50 | 0.0852 |

The model converges cleanly from ~7.6 down to ~0.09 over 50 epochs.

---

## ✅ Predictions

```
What is the capital of France?             → paris
what is H20 commonly called?               → water
Which planet is know as the red planet?    → mars
Which bird cannot fly?                     → ostrich
What is the capital of Japan?              → tokyo
```

All 5 test predictions correct.

---

## 📁 Project Structure

```
rnn-gk-qa/
├── RNN_GK.ipynb           # Main notebook (Colab)
├── gk_qna_dataset.csv     # Custom Q&A dataset
└── README.md
```

---

## 🔧 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/rnn-gk-qa.git
cd rnn-gk-qa
```

**2. Install dependencies**
```bash
pip install torch pandas
```

**3. Open the notebook**

Run `RNN_GK.ipynb` in Google Colab or Jupyter. Upload `gk_qna_dataset.csv` to `/content/` if using Colab.

**4. Run all cells**

The notebook runs end-to-end: data loading → tokenization → vocab building → training → prediction.

---

## 🔍 Key Concepts Demonstrated

- **Custom tokenizer** — regex-based lowercasing, punctuation removal, whitespace normalization
- **Vocabulary building** — token-to-index mapping with `<unk>` handling for OOV tokens
- **PyTorch Dataset & DataLoader** — custom `QADataset` class with variable-length sequences
- **RNN forward pass** — using only the final hidden state for classification
- **Training loop** — manual gradient descent with Adam optimizer and CrossEntropyLoss
- **Inference** — `predict()` function with `model.eval()` and `torch.no_grad()`

---

## ⚠️ Limitations

- Answers are predicted as **single tokens only** — multi-word answers (e.g., "Alexander Graham Bell") are truncated to the first token
- No padding/masking — batch size is fixed at 1 due to variable sequence lengths
- Dataset is small (171 samples) and closed-domain — will fail on unseen question patterns
- Simple RNN, not LSTM/GRU — prone to vanishing gradients on longer sequences

---

## 🚀 Possible Improvements

- [ ] Pad sequences and train with batch_size > 1
- [ ] Support multi-token answer prediction (sequence-to-sequence)
- [ ] Replace SimpleRNN with LSTM or GRU
- [ ] Expand dataset for better generalization
- [ ] Add train/val/test split with accuracy metrics

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow)

---

## 👤 Author

**Arman** — Final-year CSE student | Research Assistant @ East Delta University Innovation Lab  
Building toward a career in AI/ML.

---

*Built as a learning project to understand RNN internals from scratch.*
