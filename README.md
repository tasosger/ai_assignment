# Exercise 2 - Machine Learning and Deep Learning

This repository contains three parts exploring different machine learning and deep learning approaches for text and image classification tasks.

## Project Structure

```
exercise2/
├── partA.ipynb          # IMDB sentiment analysis with traditional ML
├── partB.ipynb          # IMDB sentiment analysis with deep learning (RNN/LSTM/GRU)
├── partC.ipynb          # Fashion-MNIST classification with transfer learning
├── imdb_train.pt        # Preprocessed IMDB training data
├── imdb_test.pt         # Preprocessed IMDB test data
└── README.md            # This file
```

## Part A: IMDB Sentiment Analysis (Traditional ML)

**Objective**: Binary sentiment classification (positive/negative) on the IMDB movie review dataset using traditional machine learning approaches.

### Key Features:
- **Data Preprocessing**:
  - Text normalization and tokenization
  - HTML tag removal (`<br />` tags)
  - Stopword removal
  - Custom tokenization with regex patterns

- **Feature Selection**:
  - Document frequency computation
  - Candidate word selection (removing rare and common words)
  - Information Gain (IG) based feature selection
  - Final vocabulary size: 5,000 features

- **Models**:
  - **AdaBoost Classifier** with Decision Tree stumps
    - Grid search over `n_estimators` and `learning_rate`
    - Best params: `learning_rate=0.05`, `n_estimators=200`
    - Test accuracy: ~60%

  - **Random Forest Classifier**
    - Grid search over `n_estimators` and `max_depth`
    - Best params: `max_depth=None`, `n_estimators=200`
    - Test accuracy: ~82%

### Dataset:
- Training set: 20,000 samples
- Validation set: 5,000 samples
- Test set: 25,000 samples
- Balanced class distribution (50% positive, 50% negative)

## Part B: IMDB Sentiment Analysis (Deep Learning)

**Objective**: Binary sentiment classification using deep learning with RNN architectures and Word2Vec embeddings.

### Key Features:
- **Word Embeddings**:
  - Word2Vec (Skip-gram) with 200-dimensional embeddings
  - Vocabulary built from training data (min frequency = 2)
  - Pre-trained embeddings combined with random initialization for OOV words

- **Model Architecture**:
  - Stacked bidirectional RNN (LSTM/GRU) with max pooling
  - Configurable layers, hidden dimensions, and dropout
  - Embedding layer initialized with Word2Vec weights

- **Experiments**:
  - **LSTM_2x256_maxpool**: 2 layers, 256 hidden units, dropout=0.3
    - Best dev accuracy: 87.04%
  - **GRU_2x256_maxpool**: 2 layers, 256 hidden units, dropout=0.3
    - Best dev accuracy: 87.92% (best model)
  - **LSTM_1x128_maxpool**: 1 layer, 128 hidden units, dropout=0.2
    - Best dev accuracy: 86.50%

- **Training**:
  - Early stopping with patience=3
  - Gradient clipping (max norm=1.0)
  - Adam optimizer with learning rate=2e-3
  - Test accuracy: ~85%

### Dataset:
- Training set: 20,000 samples
- Validation set: 5,000 samples
- Test set: 24,999 samples (after filtering empty sequences)

## Part C: Fashion-MNIST Classification (Transfer Learning)

**Objective**: Multi-class classification of Fashion-MNIST images using transfer learning with ResNet18.

### Key Features:
- **Dataset**: Fashion-MNIST (10 classes)
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation (±10 degrees)
  - Resize to 224x224 (for ResNet input)
  - Normalization

- **Model Architecture**:
  - **Base Model**: ResNet18 pre-trained on ImageNet
  - **Adaptation**:
    - GrayToRGB layer to convert grayscale to RGB
    - Frozen ResNet18 backbone (feature extractor)
    - Custom classification head:
      - 3 fully connected layers (512 hidden units each)
      - ReLU activations
      - Output layer for 10 classes

- **Training**:
  - Adam optimizer (lr=0.001)
  - CrossEntropyLoss
  - 5 epochs
  - Final test accuracy: ~85%

### Results:
- Best development accuracy: ~85%
- Balanced performance across all 10 classes
- Highest precision/recall for: Trouser, Sandal, Bag, Ankle boot
- Lower performance for: Shirt (most challenging class)

## Requirements

### Python Packages:
- `torch` - PyTorch for deep learning
- `torchtext==0.6.0` - Text data utilities
- `torchvision` - Vision datasets and models
- `scikit-learn` - Traditional ML models and utilities
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `nltk` - Natural language processing
- `gensim` - Word2Vec embeddings
- `matplotlib` - Plotting
- `scipy` - Sparse matrices

### Installation:
```bash
pip install torch torchtext==0.6.0 torchvision scikit-learn numpy pandas nltk gensim matplotlib scipy
```

### NLTK Data:
```python
import nltk
nltk.download('stopwords')
```

## Usage

Each part is self-contained in its respective Jupyter notebook:

1. **Part A**: Run cells sequentially to perform feature selection and train ML models
2. **Part B**: Run cells to train Word2Vec embeddings and RNN models
3. **Part C**: Run cells to download Fashion-MNIST, adapt ResNet18, and train the classifier

## Notes

- The IMDB dataset will be automatically downloaded on first run (Part A and B)
- Preprocessed data is saved to `imdb_train.pt` and `imdb_test.pt` for faster subsequent runs
- Fashion-MNIST is downloaded to `~/.pytorch/F_MNIST_data/` on first run
- All models use random seeds for reproducibility (random_state=42)

## Results Summary

| Part | Task | Best Model | Test Accuracy |
|------|------|------------|---------------|
| A | IMDB Sentiment (ML) | Random Forest | ~82% |
| B | IMDB Sentiment (DL) | GRU 2x256 | ~85% |
| C | Fashion-MNIST | ResNet18 (Transfer) | ~85% |
