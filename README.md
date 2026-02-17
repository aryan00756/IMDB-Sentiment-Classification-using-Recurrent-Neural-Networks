# IMDB Sentiment Classification using Recurrent Neural Networks

## Overview
This project implements a Deep Learning–based Binary Sentiment Classification model using the IMDB movie reviews dataset.  
The model classifies reviews as **Positive (1)** or **Negative (0)** using a Recurrent Neural Network (RNN) architecture built with TensorFlow and Keras.

---

## Problem Statement
Natural Language Processing (NLP) tasks such as sentiment analysis require understanding sequential patterns in text data.  
This project explores how Recurrent Neural Networks process sequential inputs to classify sentiment from raw text reviews.

---

## Dataset
- Dataset: IMDB Movie Reviews
- Total Samples: 50,000
- Training Samples: 25,000
- Testing Samples: 25,000
- Vocabulary Size: Top 10,000 most frequent words
- Task Type: Binary Classification

---

## Model Architecture

```
Embedding Layer
→ SimpleRNN Layer
→ Dense Output Layer (Sigmoid Activation)
```

### Architecture Details
- Embedding dimension: 128
- RNN units: 128
- Output activation: Sigmoid
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Evaluation metric: Accuracy

---

## Project Structure

```
.
├── simplernn.ipynb
├── requirements.txt
├── README.md
```

---

## Installation
Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the notebook:

```bash
jupyter notebook simplernn.ipynb
```

Or convert to script and run:

```bash
python model.py
```


---

## Key Learnings

- Understanding sequence modeling using RNN
- Text preprocessing and sequence padding
- Vocabulary limitation for computational efficiency
- Binary classification using sigmoid activation
- Model evaluation using confusion matrix and accuracy score

---

## Limitations

- SimpleRNN suffers from vanishing gradient issues
- Does not use pretrained embeddings
- No hyperparameter tuning
- No comparison with LSTM or GRU architectures

---

## Future Improvements

- Compare RNN vs LSTM vs GRU
- Integrate pretrained embeddings (GloVe)
- Hyperparameter tuning with KerasTuner
- Add F1-score, Precision, Recall, ROC-AUC metrics
- Deploy using Streamlit or FastAPI
- Add TensorBoard visualization

---

## Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn

---

## Author

Aryan Yadav

---
