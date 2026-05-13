# Cyberbullying Detection using DeBERTa and BiLSTM

An NLP-based deep learning project that detects cyberbullying content from textual data using a hybrid architecture combining **DeBERTa embeddings** and **Bidirectional LSTM networks**. The system leverages transformer-based contextual embeddings for semantic understanding and sequential learning for accurate binary text classification.

---

## 🚀 Features

* Cyberbullying text classification using Deep Learning
* Context-aware text embeddings using **DeHateBERT**
* Bidirectional LSTM for sequential feature learning
* Automatic text preprocessing and tokenization
* Binary classification:

  * `Cyberbullying`
  * `Not Cyberbullying`
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC-AUC Curve
* Visualization of training metrics and ROC curve

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **TensorFlow / Keras**
* **Transformers (Hugging Face)**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-Learn**
* **NLTK**

---

## 🧠 Model Architecture

### 1. Text Embedding Layer

* Pretrained Transformer Model:

  * `Hate-speech-CNERG/dehatebert-mono-english`
* Tokenization using Hugging Face Tokenizer
* Contextual embeddings extracted from hidden states

### 2. Deep Learning Classifier

* Bidirectional LSTM Layer
* Dropout Regularization
* Dense Neural Layers
* Sigmoid Activation for binary classification

---

## 📂 Project Workflow

1. Load and preprocess dataset
2. Remove null values
3. Tokenize text using DeHateBERT tokenizer
4. Generate contextual embeddings
5. Train BiLSTM classifier on embeddings
6. Evaluate model performance
7. Generate:

   * Classification Report
   * ROC Curve
   * Training Accuracy & Loss Graphs

---

## 📊 Dataset

The dataset contains labeled textual data classified into:

* `cyberbullying`
* `not_cyberbullying`

Example:

| Text                      | Label             |
| ------------------------- | ----------------- |
| Offensive/Harmful Comment | cyberbullying     |
| Normal Text               | not_cyberbullying |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python final.py
```

---

## 📈 Evaluation Metrics

The project evaluates performance using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Visual outputs include:

* Training Loss Graph
* Training Accuracy Graph
* ROC Curve

---

## 📁 Project Structure

```bash
├── final.py
├── cleaned_data.csv
├── requirements.txt
├── README.md
└── outputs/
```

---

## 🔥 Key Highlights

* Uses transformer-based contextual embeddings instead of traditional TF-IDF
* Combines semantic understanding with sequential learning
* Efficient hybrid NLP architecture for toxic text detection
* Suitable for:

  * Social Media Moderation
  * Online Safety Systems
  * Comment Filtering
  * Hate Speech Detection

---

## 🌍 Applications

* Social Media Monitoring
* Online Community Moderation
* AI Content Filtering
* Educational Platforms
* Mental Health & Safety Systems

---


