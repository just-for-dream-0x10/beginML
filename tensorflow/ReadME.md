# TensorFlow Learning Notes

This directory contains learning notes and example codes for the TensorFlow deep learning framework, covering various aspects from fundamental concepts to advanced applications.

## Directory Structure

- **001_basic_tensor.py**: Basic TensorFlow operations and tensor concepts
- **dpl_tensorflow.py**: Basic deep learning model implementations
- **cnn/**: Convolutional Neural Network implementations
  - Basic CNN models
  - Model improvement techniques
  - Transfer learning examples
- **RNN/**: Recurrent Neural Network implementations
  - Text generation
  - Sentiment analysis (many-to-one)
  - Sequence labeling (many-to-many)
- **GNN/**: Graph Neural Network basics
- **NLP/**: Natural Language Processing applications
  - Word embedding creation
  - Text classification models
- **GANs/**: Generative Adversarial Network implementations
- **probabilistic_tf/**: Probabilistic models and Bayesian methods
  - Hidden Markov Models
  - Dirichlet Process Mixture Models
- **linear/**: Linear models and basic regression
- **notes/**: Theoretical notes and concept explanations

## Learning Path Recommendation

1. **Getting Started**:
   - Begin with `001_basic_tensor.py` to understand basic TensorFlow operations
   - Read theoretical notes in the `notes/` directory to grasp core concepts

2. **Deep Learning Fundamentals**:
   - Study the multilayer perceptron implementation in `dpl_tensorflow.py`
   - Explore basic models in the `linear/` directory

3. **Advanced Models**:
   - CNN: Research convolutional neural network implementations in `cnn/`
   - RNN: Learn about recurrent neural network applications in `RNN/`
   - NLP: Explore natural language processing techniques in `NLP/`

4. **Advanced Applications**:
   - Generative Models: Study Generative Adversarial Networks in `GANs/`
   - Graph Models: Research Graph Neural Networks in `GNN/`
   - Probabilistic Models: Explore Bayesian methods in `probabilistic_tf/`

## Runtime Environment

The code in this project is developed based on TensorFlow 2.x, and Python 3.8+ environment is recommended. Please refer to the `requirements.txt` file for detailed dependencies.

## Running Examples

```bash
# Run basic tensor operations example
python 001_basic_tensor.py

# Run CNN example
python cnn/005_CNN.py

# Run RNN text generation example
python RNN/001_alice_text_generate.py
```
