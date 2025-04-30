# Adversarial Debiasing

## Overview

Adversarial Debiasing is an in-processing fairness technique that leverages adversarial learning to remove unwanted bias from machine learning models. The approach uses a two-network architecture: a predictor network that learns to predict the target variable, and an adversary network that attempts to predict the sensitive attribute from the predictor's internal representations.

## What Problem Does It Solve?

Adversarial Debiasing addresses the challenge of creating fair classifiers by ensuring that the model's internal representations do not contain information that could be used to discriminate based on protected attributes. It aims to achieve fairness by making it impossible for an adversary to predict the sensitive attribute from the model's learned features, while still maintaining high accuracy on the main prediction task.

## Key Concepts

- **Adversarial Learning**: A technique where two networks compete against each other during training;
- **Gradient Reversal Layer**: A special layer that reverses gradients during backpropagation;
- **Multi-objective Optimization**: Balancing the competing goals of prediction accuracy and fairness;
- **Representation Learning**: Learning internal features that are both predictive and fair.

## How It Works

1. **Network Architecture**:
   - **Predictor Network**: A standard neural network that learns to predict the target variable;
   - **Adversary Network**: A neural network that attempts to predict the sensitive attribute from the predictor's internal representations;
   - **Gradient Reversal Layer**: Connects the two networks and reverses gradients during backpropagation.

2. **Training Process**:
   - The predictor is trained to minimize prediction loss on the target variable;
   - The adversary is trained to predict the sensitive attribute from the predictor's representations;
   - The gradient reversal layer ensures that the predictor learns to make its representations useless for predicting the sensitive attribute.

3. **Hyperparameters**:
   - `lambda_adv`: Controls the strength of the adversarial component (higher values prioritize fairness over accuracy);
   - Learning rates for both networks;
   - Network architecture parameters (hidden dimensions, dropout rates, etc.).

## Implementation Details

The FairLib implementation uses PyTorch and includes:

- `GradientReversalFunction`: A custom autograd function that reverses gradients during backpropagation;
- `Predictor`: The main prediction network with batch normalization and dropout for regularization;
- `Adversary`: The adversarial network that tries to predict sensitive attributes;
- `AdversarialDebiasingModel`: Combines the predictor and adversary with training logic.

## Usage Example

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from fairlib.inprocessing.adversarial_debiasing import (
    Predictor, Adversary, AdversarialDebiasingModel
)

# Prepare your data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
a_train_tensor = torch.from_numpy(a_train.astype(np.int64))  # sensitive attribute

# Create dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, a_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model components
predictor = Predictor(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
adversary = Adversary(input_dim=64, hidden_dim=32, sensitive_dim=1)

# Create and train the adversarial debiasing model
model = AdversarialDebiasingModel(predictor=predictor, adversary=adversary, lambda_adv=1.0)
model.fit(train_loader, num_epochs=50, lr=0.001)

# Make predictions
model.eval()
X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))
with torch.no_grad():
    logits = model(X_test_tensor)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
```

## Advantages and Limitations

### Advantages
- Directly optimizes for both prediction accuracy and fairness during training;
- Can achieve better fairness-accuracy trade-offs than preprocessing methods;
- Does not require sensitive attributes during inference;

### Limitations
- Requires careful tuning of hyperparameters, especially the adversarial weight;
- Training can be unstable due to the adversarial component;
- Computationally more expensive than simpler approaches;
- May not completely eliminate all forms of bias.

## References

B. H. Zhang, B. Lemoine, and M. Mitchell, “Mitigating Unwanted Biases with Adversarial Learning,” AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.