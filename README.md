# Sentiment Analysis Using Recurrent Neural Networks (RNNs)

## Overview
This project focuses on sentiment analysis using Recurrent Neural Networks (RNNs) to classify movie reviews from the IMDB dataset and evaluate their generalization on Metacritic reviews. The notebook explores different RNN architectures, hyperparameter tuning, and performance evaluation.

## Key Features
- **Data Preprocessing**: Cleaning, tokenization, and padding of text data.
- **Model Architectures**:
  - Basic RNN
  - Tuned RNN with hyperparameter optimization
  - Bidirectional RNN with regularization
- **Evaluation**: Metrics include accuracy, precision, recall, and confusion matrices.
- **Generalization Testing**: Performance assessment on an external Metacritic dataset.

## Results
| Model              | Training Accuracy | Validation Accuracy | Test Accuracy | External Accuracy |
|--------------------|-------------------|---------------------|---------------|-------------------|
| Basic RNN          | ~100%             | ~100%               | 100%          | 48.33%            |
| Tuned RNN          | ~50%              | ~50%                | 50%           | 43.33%            |
| Bidirectional RNN  | ~50%              | ~50%                | 50%           | 43.33%            |

## Key Observations
- Basic RNN achieved perfect accuracy on IMDB but failed to generalize (overfitting).
- Hyperparameter tuning and Bidirectional RNN showed no significant improvement.
- Models struggled with negative reviews in external datasets.

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- Pandas, Numpy
- Matplotlib for visualization

## How to Use
1. Clone the repository.
2. Install required dependencies (`tensorflow`, `keras-tuner`, etc.).
3. Run the Jupyter notebook to train models and evaluate performance.

## Future Improvements
- Experiment with advanced architectures (LSTMs, GRUs, Transformers).
- Address class imbalance and dataset bias.
- Implement continuous monitoring and retraining.

## Contributors
- Anirudh Gupta (055003)
