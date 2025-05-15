# Fairness Metrics in FairLib

FairLib implements several key fairness metrics to evaluate and quantify bias in machine learning models. This documentation explains each metric, its purpose, interpretation, and typical value ranges.

## Statistical Parity Difference (SPD)

### Overview

Statistical Parity Difference measures the difference in the probability of a favorable outcome between the privileged and unprivileged groups. It quantifies whether a model's predictions are distributed equally across different demographic groups.

### What Problem Does It Solve

SPD addresses group fairness by ensuring that the overall acceptance rates for different demographic groups are similar, regardless of the ground truth. It helps identify if a model systematically favors one group over another in its predictions.

### Key Concepts

- **Statistical Parity**: Achieved when the probability of a favorable outcome is the same for all demographic groups
- **Privileged Group**: The demographic group that historically receives favorable treatment
- **Unprivileged Group**: The demographic group that historically receives unfavorable treatment

### How It Works

The Statistical Parity Difference is calculated as:

```
SPD = P(Ŷ=1|S=privileged) - P(Ŷ=1|S=unprivileged)
```

Where:
- Ŷ is the predicted outcome
- S is the sensitive attribute
- P(Ŷ=1|S=privileged) is the probability of a favorable outcome for the privileged group
- P(Ŷ=1|S=unprivileged) is the probability of a favorable outcome for the unprivileged group

### Interpretation and Value Range

- **Range**: [-1, 1]
- **Perfect Fairness**: 0 (equal probability of favorable outcomes across groups)
- **Positive Values**: Indicate the privileged group has a higher probability of favorable outcomes
- **Negative Values**: Indicate the unprivileged group has a higher probability of favorable outcomes
- **Acceptable Range**: Generally, values between -0.1 and 0.1 are considered relatively fair

### Implementation Details

In FairLib, Statistical Parity Difference is implemented as both a standalone function and a class that extends the base `Metric` class. The implementation supports multiple target and sensitive attribute columns.

### Usage Example

```python
import pandas as pd
from fairlib.metrics import StatisticalParityDifference

# Prepare your data
df = pd.DataFrame({
    'target': [1, 0, 1, 0, 1, 0, 1, 0],
    'sensitive_attribute': [0, 0, 0, 0, 1, 1, 1, 1]
})

# Set target and sensitive columns
df.targets = ['target']
df.sensitive = ['sensitive_attribute']

# Calculate Statistical Parity Difference
spd = df.statistical_parity_difference()
print(spd)
```

## Disparate Impact (DI)

### Overview

Disparate Impact measures the ratio of the probability of a favorable outcome for the unprivileged group to the probability of a favorable outcome for the privileged group. It is inspired by the "80% rule" in US anti-discrimination law.

### What Problem Does It Solve

DI helps identify indirect discrimination, where seemingly neutral practices disproportionately affect protected groups. It provides a relative measure of fairness that is often used in legal contexts.

### Key Concepts

- **Disparate Impact**: Occurs when a practice adversely affects one protected group more than another
- **80% Rule**: A legal guideline stating that the selection rate for any protected group should be at least 80% of the rate for the group with the highest selection rate

### How It Works

The Disparate Impact is calculated as:

```
DI = P(Ŷ=1|S=unprivileged) / P(Ŷ=1|S=privileged)
```

Where:
- Ŷ is the predicted outcome
- S is the sensitive attribute
- P(Ŷ=1|S=unprivileged) is the probability of a favorable outcome for the unprivileged group
- P(Ŷ=1|S=privileged) is the probability of a favorable outcome for the privileged group

### Interpretation and Value Range

- **Range**: [0, ∞)
- **Perfect Fairness**: 1.0 (equal probability of favorable outcomes across groups)
- **Values < 1**: Indicate the unprivileged group has a lower probability of favorable outcomes
- **Values > 1**: Indicate the unprivileged group has a higher probability of favorable outcomes
- **Acceptable Range**: Generally, values between 0.8 and 1.25 are considered relatively fair, with the legal threshold being 0.8 (the "80% rule")

### Implementation Details

In FairLib, Disparate Impact is implemented as both a standalone function and a class that extends the base `Metric` class. The implementation supports multiple target and sensitive attribute columns.

### Usage Example

```python
import pandas as pd
from fairlib.metrics import DisparateImpact

# Prepare your data
df = pd.DataFrame({
    'target': [1, 0, 1, 0, 1, 0, 1, 0],
    'sensitive_attribute': [0, 0, 0, 0, 1, 1, 1, 1]
})

# Set target and sensitive columns
df.targets = ['target']
df.sensitive = ['sensitive_attribute']

# Calculate Disparate Impact
di = df.disparate_impact()
print(di)
```

## Equality of Opportunity (EOO)

### Overview

Equality of Opportunity measures the difference in true positive rates between privileged and unprivileged groups. It focuses on ensuring that individuals who qualify for a favorable outcome have an equal chance of being correctly classified, regardless of their protected attribute.

### What Problem Does It Solve

EOO addresses a specific type of fairness concern: ensuring that qualified individuals from different groups have equal chances of receiving positive outcomes. It is particularly relevant in scenarios where false negatives have significant consequences, such as loan approvals or hiring decisions.

### Key Concepts

- **True Positive Rate (TPR)**: The proportion of actual positives that are correctly identified
- **Equal Opportunity**: Achieved when TPRs are equal across different demographic groups

### How It Works

The Equality of Opportunity is calculated as the difference in true positive rates between privileged and unprivileged groups:

```
EOO = TPR_privileged - TPR_unprivileged
```

Where:
- TPR_privileged = P(Ŷ=1|Y=1, S=privileged) is the true positive rate for the privileged group
- TPR_unprivileged = P(Ŷ=1|Y=1, S=unprivileged) is the true positive rate for the unprivileged group

This measures whether qualified individuals (Y=1) have equal chances of receiving positive predictions (Ŷ=1) regardless of their protected attribute status.

### Interpretation and Value Range

- **Range**: [-1, 1]
- **Perfect Fairness**: 0 (equal true positive rates across groups)
- **Positive Values**: Indicate the privileged group has a higher true positive rate
- **Negative Values**: Indicate the unprivileged group has a higher true positive rate
- **Acceptable Range**: Generally, values between -0.1 and 0.1 are considered relatively fair

### Implementation Details

In FairLib, Equality of Opportunity is implemented as both a standalone function and a class that extends the base `Metric` class. The implementation requires predictions in addition to the target and sensitive attribute columns.



## Choosing the Right Metric

The choice of fairness metric depends on the specific context and fairness goals:

- **Statistical Parity Difference**: Use when you want to ensure equal representation across groups, regardless of qualifications.
- **Disparate Impact**: Use when you need a relative measure that aligns with legal standards like the 80% rule.
- **Equality of Opportunity**: Use when you want to ensure that qualified individuals have equal chances of receiving positive outcomes, regardless of their protected attributes.

It's important to note that different fairness metrics may be in tension with each other, and it's often impossible to satisfy all fairness criteria simultaneously. The choice of metric should be guided by the specific ethical and practical considerations of your application.

## Additional Resources

For practical examples of using these metrics, refer to the Jupyter notebooks in the `examples/` directory of the FairLib package.

## References

1. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference.
2. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In Advances in neural information processing systems.
3. Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining.