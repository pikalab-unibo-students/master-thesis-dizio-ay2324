# Reweighing

## Overview

Reweighing is a preprocessing technique for fairness in machine learning that works by assigning different weights to different instances in the training data. The goal is to ensure that the combination of class and protected attribute values becomes statistically independent.

## What Problem Does It Solve?

Reweighing addresses the problem of statistical discrimination in training data. When the distribution of outcomes differs across protected groups, models trained on this data may perpetuate or amplify these disparities. Reweighing mitigates this issue by adjusting the importance of each training instance to achieve statistical parity.

## Key Concepts

- **Instance Weights**: Each training instance is assigned a weight that determines its importance during model training;
- **Statistical Independence**: The algorithm aims to make the protected attribute and the outcome statistically independent;
- **Weight Calculation**: Weights are calculated to balance the representation of these four groups.

## How It Works

1. **Group Division**:
   - Divide the training data into four groups based on protected attribute and outcome:
     - Privileged and favorable outcome;
     - Privileged and unfavorable outcome;
     - Unprivileged and favorable outcome;
     - Unprivileged and unfavorable outcome.

2. **Weight Calculation**:
   - For each group, calculate a weight using the formula:
     ```
     weight = (expected_proportion / observed_proportion)
     ```
   - Where:
     - `expected_proportion` = (proportion of instances with this outcome) × (proportion of instances with this protected attribute value);
     - `observed_proportion` = (proportion of instances in this specific group).

3. **Application**:
   - Assign the calculated weights to each instance in the training data;
   - Use these weights during model training (most ML algorithms support sample weights).

## Implementation Details

FairLib provides two implementations of Reweighing:

1. **Reweighing**: The standard implementation that works with a single sensitive attribute

2. **ReweighingWithMean**: An extended implementation that can handle multiple sensitive attributes by:
   - Calculating weights for each sensitive attribute independently;
   - Taking the mean of these weights for each instance.

Both implementations extend the `Transformer` class and implement the `transform` method that adds a "weights" column to the input DataFrame.

## Usage Example

```python
import pandas as pd
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression

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

# Apply reweighing
reweighing = Reweighing()
transformed_df = reweighing.transform(df)

# Extract features, target, and weights
X = transformed_df[["feature1", "feature2"]]
y = transformed_df["target"]
weights = transformed_df["weights"]

# Train a model using the weights
model = LogisticRegression()
model.fit(X, y, sample_weight=weights)
```

## Multiple Sensitive Attributes

For multiple sensitive attributes, use `ReweighingWithMean`:

```python
from fairlib.preprocessing import ReweighingWithMean

# Specify multiple sensitive columns
df.sensitive = {"gender", "race"}

# Apply reweighing with mean
reweighing = ReweighingWithMean()
transformed_df = reweighing.transform(df)
```

## Advantages and Limitations

### Advantages
- Simple and intuitive approach to fairness;
- Compatible with most machine learning algorithms that support sample weights;
- Preprocessing approach that doesn't require modifying the learning algorithm;
- Can handle multiple sensitive attributes (with ReweighingWithMean).

### Limitations
- May not be effective if the training data has very few samples in some groups;
- Requires sensitive attribute information during training (but not during inference);

## References

F. Kamiran and T. Calders, “Data Preprocessing Techniques for Classification without Discrimination,” Knowledge and Information Systems, 2012.