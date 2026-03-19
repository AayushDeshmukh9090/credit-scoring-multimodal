# Multi-Modal Credit Scoring (ANN + Feature Fusion)

A dual-branch neural network for credit default prediction using the 
Home Credit Default Risk dataset.

## Architecture
- Branch 1: Numerical features → Dense layers
- Branch 2: Categorical features → Entity Embeddings
- Fusion: Concatenation → Decision head
- Fairness: Demographic parity evaluation on gender

## Tech Stack
Python, TensorFlow/Keras, FastAPI, Streamlit, scikit-learn, pandas

## Status
🚧 In Progress