# Prejudice Remover

## Overview

The Prejudice Remover is an in-processing fairness algorithm that learns a classifier while removing direct and indirect prejudice by adding a regularization term to the objective function.

## What Problem Does It Solve?

Prejudice Remover addresses the problem of both direct and indirect discrimination in machine learning models. Direct discrimination occurs when decisions are explicitly based on sensitive attributes, while indirect discrimination happens when decisions are based on seemingly neutral attributes that are correlated with sensitive attributes. The algorithm aims to minimize the mutual information between the sensitive attribute and the prediction, ensuring that the model's decisions are not influenced by protected characteristics.

## Key Concepts

- **Mutual Information**: A measure of the statistical dependence between two random variables (in this case, between the sensitive attribute and the prediction);
- **Prejudice Regularizer**: A penalty term added to the loss function that reduces the mutual information between sensitive attributes and predictions;
- **Eta Parameter**: Controls the strength of the fairness regularization (higher values prioritize fairness over accuracy);
- **In-processing Approach**: Works by modifying the learning algorithm itself rather than preprocessing the data or post-processing the predictions.


## Implementation Details

FairLib implements the Prejudice Remover algorithm using PyTorch, with the following components:

1. **PrejudiceRemoverLoss**: A custom loss function that:
   - Takes a base loss function (default: BCELoss);
   - Adds a regularization term based on mutual information;
   - Calculates joint and marginal probabilities for sensitive attributes and predictions;
   - Computes the KL divergence as a measure of mutual information.

2. **PrejudiceRemover**: The main class that:
   - Wraps a PyTorch model;
   - Uses the custom loss function during training;
   - Handles data preparation, including sensitive attribute extraction;
   - Implements fit and predict methods compatible with the FairLib interface.

## Usage Example

```python
import torch
import torch.nn as nn
from fairlib import DataFrame
from fairlib.inprocessing import PrejudiceRemover

# Prepare your data
df = DataFrame({
    "feature1": [...],
    "feature2": [...],
    "target": [...],      # Target variable (0 or 1)
    "sensitive": [...]   # Sensitive attribute (0 or 1)
})

# Specify target and sensitive columns
df.targets = "target"
df.sensitive = "sensitive"

# Create a PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = 2  # Number of features
model = SimpleModel(input_dim)

# Create and train the Prejudice Remover
prejudice_remover = PrejudiceRemover(
    torchModel=model,
    eta=1.0  # Fairness regularization strength
)

# Train the model
prejudice_remover.fit(df, epochs=100, batch_size=32)

# Make fair predictions
predictions = prejudice_remover.predict(df)
```

## Advantages and Limitations

### Advantages

- **In-processing Approach**: Directly addresses fairness during model training;
- **Flexibility**: Works with any differentiable model architecture;
- **Tunable Fairness**: The eta parameter allows for adjustable fairness-accuracy trade-offs;
- **Handles Both Direct and Indirect Discrimination**: Addresses both explicit and implicit biases.

### Limitations

- **Computational Complexity**: More computationally intensive than pre-processing approaches;
- **Binary Sensitive Attributes**: Current implementation works best with binary sensitive attributes;
- **Requires PyTorch**: Limited to PyTorch models, not compatible with other ML frameworks;
- **Hyperparameter Sensitivity**: Performance can be sensitive to the choice of eta;
- **Training Stability**: May require careful tuning of learning rates and other training parameters.

## References

T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma, “Fairness-Aware Classifier with Prejudice Remover Regularizer,” Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2012.