import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Dipendenze
# ----------------------------------------------------------------------------
import openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

from fairlib import DataFrame
from fairlib.inprocessing import AdversarialDebiasing, Fauci, PrejudiceRemover

# ----------------------------------------------------------------------------
# Parametri globali
# ----------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 64
LAMBDA_ADV = 1.5
REGULARIZATION_WEIGHT = 0.6
ETA = 0.6

# ----------------------------------------------------------------------------
# Funzioni di supporto
# ----------------------------------------------------------------------------

def prepare_adult_dataset() -> DataFrame:
    adult_ds = openml.datasets.get_dataset(179)
    adult_df, *_ = adult_ds.get_data(dataset_format="dataframe")
    adult_df.rename(columns={"class": "income"}, inplace=True)
    adult_df.drop(columns=["fnlwgt"], inplace=True)

    adult = DataFrame(adult_df)
    adult.targets, adult.sensitive = "income", ["sex"]  # sempre lista

    for col in adult.columns:
        if adult[col].dtype == "object" or adult[col].dtype.name == "category":
            adult[col], _ = pd.factorize(adult[col])
    return adult


def evaluate_fairness(X_test, y_pred, positive_target=1, favored_class=0):
    """
    Evaluate the fairness metrics (SPD and DI) of the predictions.
    The positive_target and favored_class parameters allow you to specify
    which target is considered positive and which is considered favored.
    """
    X_test = X_test.copy()
    X_test["income"] = y_pred
    dataset = DataFrame(X_test)
    dataset.targets = "income"
    dataset.sensitive = ["sex"]  # omogeneo

    spd = dataset.statistical_parity_difference()[{"income": positive_target, "sex": favored_class}]
    di = dataset.disparate_impact()[{"income": positive_target, "sex": favored_class}]
    return spd, di


def train_baseline_classifier(X: DataFrame, y: pd.Series):
    """Train a baseline neural network classifier using SimpleNet"""
    model = SimpleNet(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Convert data to tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.values)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

# ----------------------------------------------------------------------------
# Funzioni di plotting
# ----------------------------------------------------------------------------

def _safe_values(values):
    return [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in values]


def _save_barplot(values, algorithms, title, ylabel, filename, ylim=None):
    values = _safe_values(values)
    sns.set_palette("husl")
    plt.style.use("seaborn-v0_8")
    colors = sns.color_palette()[: len(algorithms)]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(algorithms, values, color=colors, alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

    for bar, raw_val in zip(bars, values):
        height = bar.get_height()
        label = "NaN" if raw_val == 0 and np.isnan(raw_val) else f"{raw_val:.3f}"
        offset = 0.005 if ylim is None else (ylim[1] - ylim[0]) * 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato: {filename}")


def save_plots(results):
    algos = [r["algorithm"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    spd_abs = [abs(r["spd"]) for r in results]
    di_vals = [r["di"] for r in results]
    di_abs = [abs(d - 1) if (d is not None and np.isfinite(d)) else np.nan for d in di_vals]

    _save_barplot(accuracy, algos, "Accuracy – In-processing Algorithms", "Accuracy", "inprocessing_accuracy_comparison.png", ylim=(0.70, 0.85))
    _save_barplot(spd_abs, algos, "|Statistical Parity Difference| – Lower is Better", "|SPD|", "inprocessing_spd_comparison.png")
    _save_barplot(di_abs, algos, "|Disparate Impact − 1| – Lower is Better", "|DI−1|", "inprocessing_di_comparison.png")


# ----------------------------------------------------------------------------
# Rete PyTorch
# ----------------------------------------------------------------------------

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    adult = prepare_adult_dataset()
    X_full = adult.drop(columns=["income"]).copy()
    y_full = adult["income"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.35, random_state=42)

    results: list[dict[str, float | str]] = []

    # --------------------------- Baseline ---------------------------
    base_clf = train_baseline_classifier(X_train, y_train)

    # Make predictions with the neural network (niente secondo sigmoid)
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values)
        base_probs = base_clf(X_test_tensor)
        base_pred = (base_probs > 0.5).int().numpy()

    base_acc = accuracy_score(y_test, base_pred)
    base_spd, base_di = evaluate_fairness(X_test, base_pred)
    results.append({"algorithm": "Baseline", "accuracy": base_acc, "spd": base_spd, "di": base_di})

    # ------------------ Adversarial Debiasing ----------------------
    ds_adv_train = DataFrame(X_train.copy()); ds_adv_train.sensitive = ["sex"]
    adv_model = AdversarialDebiasing(input_dim=X_train.shape[1], hidden_dim=32, output_dim=1, sensitive_dim=1, lambda_adv=LAMBDA_ADV)
    adv_model.fit(ds_adv_train, y_train.values, num_epochs=EPOCHS, batch_size=BATCH_SIZE)

    ds_adv_test = DataFrame(X_test.copy()); ds_adv_test.sensitive = ["sex"]
    adv_pred = adv_model.predict(ds_adv_test).cpu().numpy().astype(int).flatten()
    adv_acc = accuracy_score(y_test, adv_pred)
    adv_spd, adv_di = evaluate_fairness(X_test, adv_pred)
    results.append({"algorithm": "AdversarialDebiasing", "accuracy": adv_acc, "spd": adv_spd, "di": adv_di})

    # --------------------------- Fauci -----------------------------
    simple_net = SimpleNet(input_dim=X_train.shape[1])
    ds_fauci_train = DataFrame(X_train.copy()); ds_fauci_train.sensitive = ["sex"]
    fauci = Fauci(torchModel=simple_net, loss=nn.BCELoss(), fairness_regularization="spd", regularization_weight=REGULARIZATION_WEIGHT)
    fauci.fit(ds_fauci_train, y_train.values, epochs=EPOCHS, batch_size=BATCH_SIZE)
    ds_fauci_test = DataFrame(X_test.copy()); ds_fauci_test.sensitive = ["sex"]
    fauci_probs = fauci.predict(ds_fauci_test)
    fauci_pred = (fauci_probs > 0.5).int().cpu().numpy()
    fauci_acc = accuracy_score(y_test, fauci_pred)
    fauci_spd, fauci_di = evaluate_fairness(X_test, fauci_pred)
    results.append({"algorithm": "Fauci", "accuracy": fauci_acc, "spd": fauci_spd, "di": fauci_di})

    # --------------------- Prejudice Remover -----------------------
    simple_net = SimpleNet(input_dim=X_train.shape[1])
    ds_pr_train = DataFrame(X_train.copy()); ds_pr_train.sensitive = ["sex"]
    pr_model = PrejudiceRemover(torchModel=simple_net, loss=nn.BCELoss(), eta=ETA)
    pr_model.fit(ds_pr_train, y_train.values, epochs=EPOCHS, batch_size=BATCH_SIZE)
    ds_pr_test = DataFrame(X_test.copy()); ds_pr_test.sensitive = ["sex"]
    pr_probs = pr_model.predict(ds_pr_test)
    pr_pred = (pr_probs > 0.5).int().cpu().numpy()
    pr_acc = accuracy_score(y_test, pr_pred)
    pr_spd, pr_di = evaluate_fairness(X_test, pr_pred)
    results.append({"algorithm": "PrejudiceRemover", "accuracy": pr_acc, "spd": pr_spd, "di": pr_di})

    # -------------------- Stampa e grafici -------------------------
    print("\n----- RISULTATI -----")
    for r in results:
        print(f"{r['algorithm']} Acc: {r['accuracy']}  SPD: {r['spd']}  DI: {r['di']}")

    save_plots(results)


if __name__ == "__main__":
    main()
