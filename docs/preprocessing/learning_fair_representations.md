# Learning Fair Representations (LFR)

## Overview

Learning Fair Representations (LFR) is a preprocessing technique for achieving fairness in machine learning models. It works by transforming the original input features into a new representation that preserves the information needed for the main prediction task while removing information about sensitive attributes.

## What Problem Does It Solve?

LFR addresses the challenge of creating fair classifiers by learning a new representation of the data that:
1. Encodes the training data as well as possible;
2. Obfuscates information about protected attributes;
3. Preserves enough information to predict the target variable accurately.

This approach allows any standard machine learning algorithm to be applied to the transformed data without explicitly incorporating fairness constraints.

## Key Concepts

- **Latent Representation**: A lower-dimensional encoding of the original features that removes sensitive information;
- **Encoder-Decoder Architecture**: Neural network components that transform data to and from the latent space;
- **Multi-objective Loss Function**: Balances three competing objectives:
  - Reconstruction loss (Lx): Ensures the latent representation preserves useful information;
  - Fairness loss (Lz): Minimizes statistical disparity between groups in the latent space;
  - Classification loss (Ly): Maintains predictive power for the target variable.

## How It Works

1. **Neural Network Architecture**:
   - **Encoder**: Transforms input features into a fair latent representation;
   - **Decoder**: Reconstructs original features from the latent representation;
   - **Classifier**: Predicts the target variable from the latent representation.

2. **Training Process**:
   - The model is trained to minimize a weighted combination of three losses:
     - Reconstruction loss: Measures how well the original features can be reconstructed;
     - Fairness loss: Measures statistical disparity between protected groups in the latent space;
     - Classification loss: Measures prediction accuracy on the target variable.

3. **Hyperparameters**:
   - `alpha_x`: Weight for the reconstruction loss component;
   - `alpha_y`: Weight for the classification loss component;
   - `alpha_z`: Weight for the fairness loss component;
   - `latent_dim`: Dimension of the fair representation (smaller than input dimension).

## Implementation Details

The FairLib implementation uses PyTorch and includes:

- `Encoder`: Neural network that transforms input features to latent representation;
- `Decoder`: Neural network that reconstructs original features from latent representation;
- `Classifier`: Neural network that predicts target variable from latent representation;
- `compute_fairness_loss`: Calculates statistical parity difference in the latent space;
- `compute_reconstruction_loss`: Measures how well original features are preserved;
- `compute_classification_loss`: Measures prediction accuracy.

## Usage Example

```python
import torch
from fairlib.preprocessing.lfr import LFR
from sklearn.model_selection import train_test_split

# Prepare your data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    features, labels, sensitive_attributes, test_size=0.2
)

# Initialize LFR model
lfr = LFR(
    input_dim=X_train.shape[1],  # Number of input features
    latent_dim=8,                # Dimension of fair representation
    output_dim=X_train.shape[1], # Same as input for reconstruction
    alpha_z=1.0,                 # Weight for fairness loss
    alpha_x=1.0,                 # Weight for reconstruction loss
    alpha_y=1.0                  # Weight for classification loss
)

# Train the model
lfr.fit(X_train, y_train, s_train, epochs=100)

# Transform data to fair representation
X_train_fair = lfr.transform(X_train)
X_test_fair = lfr.transform(X_test)

# Use transformed data with any classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train_fair, y_train)
predictions = clf.predict(X_test_fair)
```

## Advantages and Limitations

### Advantages
- Creates a reusable fair representation that can be used with any classifier;
- Balances multiple objectives: fairness, accuracy, and data fidelity;
- Provides control over the fairness-utility trade-off through hyperparameters.

### Limitations
- Requires sensitive attribute information during training (but not during inference);
- Finding optimal hyperparameters can be challenging;
- Neural network training may require significant computational resources.

## References

R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations.” International Conference on Machine Learning, 2013.