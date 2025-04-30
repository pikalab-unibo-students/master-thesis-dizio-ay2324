# Disparate Impact Remover

## Overview

The Disparate Impact Remover is a fairness-aware preprocessing algorithm that aims to remove disparate impact by transforming feature distributions to a common (median) distribution across sensitive attribute groups. 

## What Problem Does It Solve?

Disparate impact occurs when a seemingly neutral practice has a disproportionate adverse effect on a protected group. The Disparate Impact Remover addresses this issue by modifying the input features in a way that makes it difficult to predict the sensitive attribute from the transformed features, while preserving as much information as possible for the main prediction task.

## Key Concepts

- **Repair Level**: Controls the degree of repair applied to the data (from 0.0 to 1.0):
  - 0.0 = no repair (original data);
  - 1.0 = full repair (maximum fairness, minimum predictability of sensitive attribute);
  - Values between 0 and 1 provide a trade-off between fairness and utility.

- **Quantile Transformation**: The algorithm works by transforming each feature's distribution within each sensitive group to match a common median distribution.


## How It Works

1. **Fit Phase**:
   - For each feature, the algorithm computes the empirical cumulative distribution function (CDF) for each sensitive group;
   - It then creates a shared "median" quantile function across groups by taking the median value at each quantile point.

2. **Transform Phase**:
   - For each feature value in a given sensitive group, the algorithm:
     - Maps the value to its corresponding rank (quantile) in that group's distribution;
     - Transforms the value by mapping the rank to the corresponding value in the median distribution;
     - Applies the repair level to control how much transformation is applied.

3. **Result**:
   - The transformed features have distributions that are more similar across sensitive groups;
   - This makes it harder to predict the sensitive attribute from the transformed features.

## Implementation Details

The implementation in FairLib uses NumPy for efficient numerical operations and includes the following key components:

- `_make_cdf`: Creates a function that maps values to their quantiles in a distribution;
- `_make_inverse_cdf`: Creates a function that maps quantiles to values;
- `_transform_feature`: Applies the quantile transformation to a single feature.

## Usage Example

```python
import numpy as np
from fairlib.preprocessing.disparate_impact_remover import DisparateImpactRemover
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Prepare your data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    features, labels, sensitive_attributes, test_size=0.2
)

# Initialize the DisparateImpactRemover with desired repair level
dir = DisparateImpactRemover(repair_level=0.8)

# Fit the transformer on training data
dir.fit(X_train, s=s_train)

# Transform both training and test data
X_train_transformed = dir.transform(X_train, s=s_train)
X_test_transformed = dir.transform(X_test, s=s_test)

# Train a classifier on the transformed data
clf = LogisticRegression()
clf.fit(X_train_transformed, y_train)

# Make predictions
y_pred = clf.predict(X_test_transformed)
```

## Advantages and Limitations

### Advantages
- Preprocessing approach that works with any downstream classifier;
- Provides a controllable trade-off between fairness and accuracy via the repair level parameter;
- Directly addresses disparate impact in the data.

### Limitations
- May reduce overall accuracy of the model;
- Requires sensitive attribute information during both training and inference;
- Works best with numerical features; categorical features need special handling.

## References

M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and S. Venkatasubramanian, “Certifying and removing disparate impact.” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015.