
# RNN Text Generation with TensorFlow
This project demonstrates how to build a character-level text generation model using Recurrent Neural Networks (RNNs) with TensorFlow and Keras. The model is trained on classic literature texts from Project Gutenberg to learn patterns in language and generate new text.

## Overview
The script uses a GRU (Gated Recurrent Unit) based model to predict the next character in a sequence. After training, it can generate new text that mimics the style of the training data.

## Features
- Downloads and preprocesses text data from Project Gutenberg
- Builds a character-level language model using GRU cells
- Implements stateful RNN for better sequence modeling
- Saves model checkpoints during training
- Generates sample text after each training phase

## Requirements
- TensorFlow 2.x
- NumPy
- Python 3.10+
- scikit-learn (for 002_many_to_one.py)
- NLTK (for 003_many_to_many.py)

## How It Works

### 001_basic.py - Text Generation
1. The script downloads text data from Project Gutenberg
2. Text is processed into character sequences
3. A GRU-based model is trained to predict the next character in a sequence
4. After every 10 epochs, the model generates sample text starting with "Alice "

### 002_many_to_one.py - Sentiment Analysis
1. Downloads and processes labeled sentiment sentences dataset
2. Builds a sentiment analysis model using Bidirectional LSTM
3. Converts sentences to sequences and applies padding
4. Trains the model for binary classification (positive/negative)
5. Evaluates model performance and outputs confusion matrix

### 003_many_to_many.py - POS Tagging
1. Downloads and processes the NLTK treebank dataset
2. Builds a Part-of-Speech (POS) tagging model using Bidirectional GRU
3. Converts sentences and POS tags to sequences and applies padding
4. Trains the model to predict POS tags for each word in a sentence
5. Evaluates model performance on test data

## Model Architecture

### 001_basic.py - Text Generation Model
The model consists of:
- An embedding layer to convert character indices to dense vectors
- A GRU (Gated Recurrent Unit) layer for sequence processing
- A dense output layer with softmax activation

### 002_many_to_one.py - Sentiment Analysis Model
The model consists of:
- An embedding layer to convert word indices to dense vectors
- A Bidirectional LSTM layer to capture contextual information
- A dense layer for feature extraction
- An output layer with sigmoid activation for binary classification

### 003_many_to_many.py - POS Tagging Model
The model consists of:
- An embedding layer to convert word indices to dense vectors
- A spatial dropout layer for regularization
- A Bidirectional GRU layer to capture contextual information
- A time-distributed dense layer to make predictions for each time step
- A softmax activation layer for multi-class classification

## Usage
Text generation model:
```bash
python ./001_basic.py
```

Sentiment analysis model:
```bash
Sentiment analysis model:
```

## Sample Output
### 001_basic.py
After training, the model will generate text samples that start with "Alice " and continue based on patterns learned from the training data. The quality of the generated text improves as training progresses.

### 002_many_to_one.py
The model will output accuracy on the test set and a confusion matrix, along with predictions for some example sentences. This demonstrates the model's ability to classify positive and negative sentiment.

## Sample Output
### 001_basic.py
After training, the model will generate text samples that start with "Alice " and continue based on patterns learned from the training data. The quality of the generated text improves as training progresses.

### 002_many_to_one.py
The model will output accuracy on the test set and a confusion matrix, along with predictions for some example sentences. This demonstrates the model's ability to classify positive and negative sentiment.

### 003_many_to_many.py
The model achieves high accuracy (>99%) on the POS tagging task after about 10 epochs of training. The model can accurately predict the part of speech for each word in a sentence, demonstrating its ability to understand syntactic structure.
