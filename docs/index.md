# FairLib Documentation

Fairlib is a Python library designed to integrate fairness-aware machine learning methods, with the goal of reducing bias in predictive models. 

## Preprocessing Techniques

Preprocessing techniques modify the training data before model training to reduce bias:

- [Disparate Impact Remover](preprocessing/disparate_impact_remover.md): Transforms feature distributions to a common median distribution across sensitive attribute groups to reduce disparate impact.
- [Learning Fair Representations (LFR)](preprocessing/learning_fair_representations.md): Uses an encoder-decoder architecture to learn a fair representation of the data that preserves predictive information while removing sensitive attribute information.
- [Reweighing](preprocessing/reweighing.md): Assigns weights to training instances to ensure statistical independence between protected attributes and outcomes.

## In-processing Techniques

In-processing techniques modify the learning algorithm to incorporate fairness constraints:

- [Adversarial Debiasing](inprocessing/adversarial_debiasing.md): Uses adversarial learning to remove information about protected attributes from the model's internal representations.
- [FaUCI (Fairness Under Constrained Injection)](inprocessing/fauci.md): Incorporates fairness metrics directly into the loss function as regularization terms.
- [Prejudice Remover](inprocessing/prejudice_remover.md): Mitigates discrimination by adding a regularization term that penalizes mutual information between predictions and sensitive attributes.

## How to Use This Documentation

Each algorithm documentation includes:

1. **Overview**: A high-level explanation of what the algorithm does
2. **What Problem Does It Solve**: The specific fairness issues addressed
3. **Key Concepts**: Important ideas and terminology
4. **How It Works**: Step-by-step explanation of the algorithm
5. **Implementation Details**: Specifics about the FairLib implementation
6. **Usage Example**: Code examples showing how to use the algorithm
7. **Advantages and Limitations**: Pros and cons of the approach
8. **References**: Academic papers and other resources

## Fairness Metrics

FairLib provides several metrics to evaluate and quantify bias in machine learning models:

- [Statistical Parity Difference](metrics.md#statistical-parity-difference-spd): Measures the difference in the probability of a favorable outcome between privileged and unprivileged groups.
- [Disparate Impact](metrics.md#disparate-impact-di): Measures the ratio of favorable outcome probabilities between unprivileged and privileged groups.
- [Equality of Opportunity](metrics.md#equality-of-opportunity-eoo): Measures the difference in true positive rates between privileged and unprivileged groups.

For detailed information about these metrics, including interpretation and typical value ranges, see the [Metrics documentation](metrics.md).

## Additional Resources

For practical examples of using these algorithms and metrics, refer to the Jupyter notebooks in the `examples/` directory of the FairLib package.