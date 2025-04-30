# FaUCI (Fairness Under Constrained Injection)

## Overview

FaUCI (Fairness Under Constrained Injection) is an in-processing fairness technique that incorporates fairness constraints directly into the model's loss function. It works by adding a fairness regularization term to the standard loss function, allowing the model to optimize for both prediction accuracy and fairness simultaneously. This approach provides a flexible framework for incorporating various fairness metrics as regularization terms.

## What Problem Does It Solve?

FAUCI addresses the challenge of creating fair machine learning models without sacrificing too much predictive performance. By incorporating fairness metrics directly into the training process, it allows for explicit control over the trade-off between accuracy and fairness. This approach is particularly useful when specific fairness constraints must be satisfied while maintaining model performance.

## Key Concepts

- **Penalized Loss Function**: Combines a standard loss function with a fairness regularization term;
- **Fairness Regularization**: Uses fairness metrics (like Statistical Parity Difference) as regularization terms;
- **Regularization Weight**: Controls the trade-off between prediction accuracy and fairness.

## How It Works

1. **Loss Function Design**:
   - The loss function is a weighted combination of a standard loss (e.g., MSE, BCE) and a fairness regularization term;
   - The formula is: `total_loss = (1 - weight) * base_loss + weight * regularizer_loss`;
   - The `weight` parameter controls the importance of fairness vs. accuracy.

2. **Fairness Metrics as Regularizers**:
   - Various fairness metrics can be used as regularization terms;
   - Common options include Statistical Parity Difference (SPD) or Disparate Impact (DI);
   - The fairness metric is computed on mini-batches during training.

3. **Training Process**:
   - The model is trained using standard gradient-based optimization;
   - Gradients flow through both the prediction loss and the fairness regularization term;
   - The model learns to make predictions that are both accurate and fair.

## Implementation Details

The FairLib implementation includes:

- `BaseLoss`: A wrapper for standard loss functions;
- `PenalizedLoss`: Extends BaseLoss to include fairness regularization;
- `Fauci`: Main class that wraps a PyTorch model and trains it with fairness constraints.

The implementation is flexible and can work with any PyTorch model and various fairness metrics.

## Usage Example

```python
import torch
import torch.nn as nn
from fairlib.inprocessing import Fauci
from fairlib import DataFrame

# Prepare your data
df = DataFrame(your_data)
df.targets = "target_column"
df.sensitive = "sensitive_column"

# Create a PyTorch model
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Create FAUCI model with Statistical Parity Difference as regularizer
fauci_model = Fauci(
    torchModel=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    loss=nn.BCELoss(),
    fairness_regularization="spd",  # Statistical Parity Difference
    regularization_weight=0.5      # Equal weight to accuracy and fairness
)

# Train the model
fauci_model.fit(df, epochs=100, batch_size=32)

# Make predictions
predictions = fauci_model.predict(test_features)
```

## Advantages and Limitations

### Advantages
- Directly optimizes for both prediction accuracy and fairness during training;
- Provides explicit control over the fairness-accuracy trade-off;
- Compatible with various fairness metrics.

### Limitations
- Requires careful tuning of the regularization weight;
- May converge to suboptimal solutions if the fairness and accuracy objectives conflict strongly;
- Computationally more expensive than simpler approaches;
- Fairness guarantees are approximate rather than strict.

## References

Magnini, M., Ciatto, G., Calegari, R., Omicini, A. (2024). Enforcing Fairness via Constraint Injection with FaUCI. Aachen : CEUR-WS.
