# AG News Text Classification using BiLSTM and Transformer (DistilBERT)

## Project Overview

This project presents a comprehensive comparison between two fundamentally different neural network architectures for document classification:

- **A Bidirectional Long Short-Term Memory (BiLSTM)** network trained entirely from scratch.
- **A pretrained Transformer (DistilBERT)** fine-tuned for downstream text classification.

Beyond comparing predictive performance, the project investigates how architectural design, tokenization strategy, transfer learning, parameter-efficient fine-tuning, and sequence length affect classification accuracy, computational efficiency, and training cost.

Several additional experiments—including recurrent model ablations, frozen encoder training, LoRA fine-tuning, and sequence-length analysis—were conducted to better understand the strengths and limitations of modern NLP architectures.

This project was completed as part of **COMP 551 – Applied Machine Learning** at **McGill University**. :contentReference[oaicite:0]{index=0}

---

# Problem Statement

The objective is to classify a news article into one of four categories:

- World
- Sports
- Business
- Sci/Tech

Unlike traditional machine learning methods that rely on handcrafted features such as TF-IDF, this project investigates **end-to-end deep learning** approaches capable of learning semantic representations directly from raw text. :contentReference[oaicite:1]{index=1}

---

# Dataset

The project uses the **AG News Corpus**, a widely used benchmark for multi-class text classification.

### Statistics

- **120,000** training articles
- **7,600** testing articles
- **4** balanced classes

A **90/10 train-validation split** was created from the official training set while keeping the official test set untouched for final evaluation.

| Category |
|-----------|
| World |
| Sports |
| Business |
| Sci/Tech | :contentReference[oaicite:2]{index=2}

---

# Methodology

The project implements two independent NLP pipelines.

## Pipeline 1 — Bidirectional LSTM

The first approach builds every component from scratch.

### Preprocessing

- Lowercase conversion
- Regular-expression tokenization
- Vocabulary construction from training data only
- Frequency thresholding
- Unknown token handling
- Dynamic padding
- Sequence truncation

Vocabulary size:

**34,306 tokens**

Maximum sequence length:

**128 tokens** :contentReference[oaicite:3]{index=3}

---

## LSTM Architecture

```
Input Text

↓

Word Tokenization

↓

Vocabulary Encoding

↓

Embedding Layer (128)

↓

Bidirectional LSTM
Hidden Size = 256

↓

Masked Mean Pooling

↓

Fully Connected Layer

↓

Softmax
```

### Why Bidirectional?

Instead of processing text only left-to-right, BiLSTM processes the sentence from both directions simultaneously, allowing each word representation to capture both previous and future context.

---

## Masked Mean Pooling

Instead of using only the final hidden state, hidden representations corresponding only to **real tokens** are averaged.

Padding tokens are ignored through masking.

Advantages:

- Better sentence representation
- Reduced sensitivity to sequence length
- Stable gradients
- Improved classification accuracy :contentReference[oaicite:4]{index=4}

---

# Transformer Pipeline

The second model fine-tunes a pretrained **DistilBERT** model.

Instead of learning language from scratch, DistilBERT begins with pretrained contextual language representations learned from millions of documents.

Pipeline:

```
Input Text

↓

DistilBERT Tokenizer

↓

Subword Encoding

↓

Attention Masks

↓

DistilBERT Encoder

↓

CLS Representation

↓

Linear Classifier

↓

Softmax
```

Unlike recurrent models, Transformers process the entire sentence simultaneously using **self-attention**, enabling efficient modeling of long-range dependencies. :contentReference[oaicite:5]{index=5}

---

# Word-Level vs Subword Tokenization

A major difference between both models lies in tokenization.

### LSTM

Word-level tokenizer

```
"Tyrannosaurus"

↓

Single Token

↓

UNK if unseen
```

Rare words may become **UNK**, causing semantic information to be lost.

---

### DistilBERT

Subword tokenizer

```
"Tyrannosaurus"

↓

Ty
##ran
##nos
##aurus
```

Rare words are decomposed into meaningful subword units, allowing the model to generalize far better to unseen vocabulary. :contentReference[oaicite:6]{index=6}

---

# Training Configuration

## BiLSTM

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Epochs | 6 |
| Batch Size | 64 |
| Hidden Size | 256 |
| Embedding Size | 128 |
| Gradient Clipping | 1.0 |
| Parameters | 5.18M |

---

## DistilBERT

| Parameter | Value |
|-----------|------:|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Gradient Clipping | 1.0 |
| Parameters | 66.96M | :contentReference[oaicite:7]{index=7}

---

# Additional Experiments

To better understand modern NLP architectures, several experiments were performed beyond the baseline implementation.

## Recurrent Model Ablation

The following recurrent architectures were compared:

- Bidirectional LSTM + Mean Pooling
- Bidirectional LSTM + Last Hidden State
- Unidirectional LSTM
- Bidirectional GRU

The BiLSTM with mean pooling consistently achieved the highest performance. :contentReference[oaicite:8]{index=8}

---

## Frozen Transformer Encoder

The pretrained DistilBERT encoder was frozen while only the classification head was trained.

Purpose:

- Reduce trainable parameters
- Faster training
- Lower GPU memory usage

Result:

Accuracy decreased significantly, demonstrating the importance of updating pretrained representations during downstream fine-tuning. :contentReference[oaicite:9]{index=9}

---

## LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) was evaluated as a parameter-efficient fine-tuning strategy.

Instead of updating every Transformer weight, LoRA learns small low-rank matrices while freezing the original model.

Benefits:

- Lower GPU memory
- Faster optimization
- Fewer trainable parameters

Although computationally efficient, LoRA produced lower accuracy than full fine-tuning under the project constraints. :contentReference[oaicite:10]{index=10}

---

## Sequence Length Analysis

Transformer models were evaluated using

- 32 tokens
- 64 tokens
- 128 tokens

Observation:

Accuracy improved from 32 to 64 tokens but plateaued beyond 64, while training time continued to increase substantially. :contentReference[oaicite:11]{index=11}

---

# Results

| Model | Test Accuracy |
|------------------------------|-------------:|
| BiLSTM | 92.29% |
| DistilBERT | **94.16%** |
| Frozen DistilBERT | 89.25% |
| LoRA DistilBERT | 89.45% |

DistilBERT achieved the highest overall accuracy, outperforming the BiLSTM by approximately **2 percentage points**, at the cost of significantly larger model size and longer training time. 

---

# Error Analysis

Both models most frequently confused

- Sci/Tech ↔ Business

These articles often discuss technology companies, products, financial markets, or IPOs, making the semantic boundary inherently ambiguous.

The Transformer produced fewer total misclassifications than the LSTM, suggesting stronger contextual understanding. 

---

# Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- NumPy
- Matplotlib
- Scikit-learn

---

# Repository Structure

```
.
├── notebooks/
├── figures/
├── report/
├── requirements.txt
└── README.md
```

---

# Key Learning Outcomes

- Sequence modeling using recurrent neural networks
- Bidirectional LSTM implementation
- Transformer fine-tuning
- Transfer learning
- Parameter-efficient fine-tuning with LoRA
- Word-level vs subword tokenization
- Attention mechanisms
- Hyperparameter optimization
- Gradient clipping
- Comparative evaluation of deep learning architectures
- Ablation studies
- Qualitative error analysis

---

# Future Improvements

Potential extensions include:

- RoBERTa and DeBERTa comparison
- BERT-base vs DistilBERT benchmarking
- Mixed precision (FP16) training
- Knowledge distillation
- Hyperparameter optimization
- Larger context windows
- Pretrained word embeddings (GloVe/FastText) for the BiLSTM
- Model ensembling

---
