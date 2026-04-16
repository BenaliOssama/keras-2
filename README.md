# Keras 2 Exercises

Neural network fundamentals using Keras on regression and multi-class classification tasks.

## Environment Setup

```bash
python3 -m venv keras_env
. keras_env/bin/activate
pip install -r requirements.txt
```

## Exercises

### Exercise 1: Regression - Optimize
Compile a regression neural network with MSE loss and MAE metrics.

**Key concepts:**
- Loss function for regression: Mean Squared Error (MSE)
- Metrics for regression: Mean Absolute Error (MAE)
- Why different problems need different loss functions

### Exercise 2: Regression Example
Train a regression network on Auto MPG dataset to predict fuel efficiency.

**What you'll do:**
- Load and preprocess Auto MPG data
- Split train/test without shuffling (80/20)
- Scale using StandardScaler (fit on train only)
- Train network with 2 hidden layers (30 neurons each)
- Achieve test MAE < 10

**Key concepts:**
- Data leakage prevention: fit scaler on train, transform both splits
- Epochs and batch_size tuning for convergence
- Evaluating on scaled data

### Exercise 3: Multi-classification - Softmax
Build a multi-class classification network architecture.

**What you'll do:**
- Create network with 2 hidden layers + softmax output layer
- Output layer neurons = number of classes (5 in example)

**Key concepts:**
- Softmax: converts logits to probabilities that sum to 1
- Output shape requirement: K neurons for K classes
- Why softmax is necessary for multi-class problems

### Exercise 4: Multi-classification - Optimize
Compile a multi-class neural network.

**Key concepts:**
- Loss function for multi-class: categorical_crossentropy
- Metrics for classification: accuracy
- Why categorical_crossentropy vs binary_crossentropy

### Exercise 5: Multi-classification Example
Train a multi-class network on Iris dataset.

**What you'll do:**
- Load Iris dataset from sklearn
- Split train/test with random_state=1
- Scale using StandardScaler
- One-hot encode labels using LabelBinarizer
- Train network with 1 hidden layer (10 neurons)
- Achieve test accuracy > 90%

**Key concepts:**
- One-hot encoding: translate categories to probability vectors
- Shape alignment: output neurons match one-hot vector length
- LabelBinarizer: automates one-hot encoding

## Core Concepts

### Regression vs Classification

**Regression:**
- Output: single continuous number (e.g., MPG)
- Loss: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE), MSE
- Output activation: linear (or none)

**Multi-class Classification:**
- Output: probability over K categories
- Loss: categorical_crossentropy
- Metrics: accuracy
- Output activation: softmax

### Loss Functions

- **MSE (Mean Squared Error):** Penalizes large errors quadratically. Used for regression.
- **MAE (Mean Absolute Error):** Linear penalty. More interpretable, less sensitive to outliers.
- **Categorical Crossentropy:** Measures difference between predicted probabilities and true one-hot labels. Used for multi-class classification.

### One-Hot Encoding

Converts categorical labels into vectors:
