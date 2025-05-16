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
   - `adv_steps`: Number of adversary updates per predictor update;
   - Learning rates for both networks;
   - Network architecture parameters (hidden dimensions, dropout rates, etc.).

## Implementation Details

The FairLib implementation uses PyTorch and includes:

- `GradientReversalFunction`: A custom autograd function that reverses gradients during backpropagation;
- `Predictor`: The main prediction network with batch normalization and dropout for regularization;
- `Adversary`: The adversarial network that tries to predict sensitive attributes;
- `AdversarialDebiasing`: Combines the predictor and adversary with training logic.

## Example Usage

```python
import fairlib as fl
from fairlib.inprocessing.adversarial_debiasing import AdversarialDebiasing

# FairLib DataFrame with one protected attribute
df = fl.DataFrame(...)
df.targets = "label"
df.sensitive = "gender"

model = AdversarialDebiasing(
    input_dim=df.shape[1],
    hidden_dim=32,
    output_dim=1,
    sensitive_dim=1,
    lambda_adv=1.0,
)

model.fit(df, num_epochs=20, batch_size=64)

y_pred = model.predict(df)  # tensor with 0/1 predictions
```

## References

B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.