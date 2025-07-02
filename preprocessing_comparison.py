"""
Fairness Comparison – LINEAR VERSION + REAL LFR + INDIVIDUAL PNG PLOTS

Esegue il confronto di fairness sul dataset Adult con quattro algoritmi di pre-processing:
  • Baseline (nessun intervento)
  • Reweighing
  • Disparate Impact Remover
  • Learning Fair Representations (LFR) – implementazione reale

Genera tre grafici (Accuracy, |SPD|, |DI−1|) e li salva come PNG.
"""

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Dipendenze
# ----------------------------------------------------------------------------
import openml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing, DisparateImpactRemover, LFR

# ----------------------------------------------------------------------------
# Funzioni di supporto
# ----------------------------------------------------------------------------

def prepare_adult_dataset():
    """Carica Adult da OpenML e applica lo stesso preprocessing di demo_core_lib.ipynb."""
    adult_ds = openml.datasets.get_dataset(179)
    adult_df, *_ = adult_ds.get_data(dataset_format="dataframe")

    adult_df.rename(columns={"class": "income"}, inplace=True)
    adult_df.drop(columns=["fnlwgt"], inplace=True)

    adult = DataFrame(adult_df)
    adult.targets, adult.sensitive = "income", ["sex"]

    # Ordinal encoding per tutte le colonne categoriche
    for col in adult.columns:
        if adult[col].dtype == "object" or adult[col].dtype.name == "category":
            adult[col], _ = pd.factorize(adult[col])
    return adult


def evaluate_fairness(X_test, y_pred, positive_target=1, favored_class=0):
    """Restituisce SPD e DI (non in valore assoluto)."""
    X_eval = X_test.copy()
    X_eval["income"] = y_pred
    ds_eval = DataFrame(X_eval)
    ds_eval.targets, ds_eval.sensitive = "income", "sex"

    spd = ds_eval.statistical_parity_difference()[{"income": positive_target, "sex": favored_class}]
    di = ds_eval.disparate_impact()[{"income": positive_target, "sex": favored_class}]
    return spd, di


def train_classifier(X, y, sample_weight=None):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf

# ----------------------------------------------------------------------------
# Funzione di plotting (una PNG per metrica)
# ----------------------------------------------------------------------------

def _save_barplot(values, algorithms, title, ylabel, filename, ylim=None):
    sns.set_palette("husl")
    plt.style.use("seaborn-v0_8")
    colors = sns.color_palette()[:len(algorithms)]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(algorithms, values, color=colors, alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

    # etichette valore
    for bar in bars:
        height = bar.get_height()
        offset = 0.005 if not ylim else (ylim[1]-ylim[0])*0.01
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                f"{height:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafico salvato: {filename}")


def save_plots(results):
    """Crea e salva i grafici Accuracy, |SPD|, |DI−1| in file PNG separati."""
    algorithms = [r['algorithm'] for r in results]

    accuracy = [r['accuracy'] for r in results]
    spd_abs = [abs(r['spd']) for r in results]
    di_abs = [abs(r['di'] - 1) for r in results]

    _save_barplot(accuracy, algorithms,
                  title="Accuracy – Pre-processing Algorithms",
                  ylabel="Accuracy",
                  filename="preprocessing_accuracy_comparison.png",
                  ylim=(0.75, 0.85))

    _save_barplot(spd_abs, algorithms,
                  title="|Statistical Parity Difference| – Lower is Better",
                  ylabel="|SPD|",
                  filename="preprocessing_spd_comparison.png")

    _save_barplot(di_abs, algorithms,
                  title="|Disparate Impact − 1| – Lower is Better",
                  ylabel="|DI−1|",
                  filename="preprocessing_di_comparison.png")

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------

def main():
    adult = prepare_adult_dataset()
    X_full = adult.drop(columns=["income"]).copy(deep=True)
    y_full = adult["income"].copy(deep=True)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.35, random_state=42)

    results = []

    # ------------------------------ Baseline ------------------------------
    base_clf = train_classifier(X_train, y_train)
    base_pred = base_clf.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_spd, base_di = evaluate_fairness(X_test, base_pred)
    results.append({"algorithm": "Baseline", "accuracy": base_acc, "spd": base_spd, "di": base_di})

    # ----------------------------- Reweighing -----------------------------
    train_rw = X_train.copy(); train_rw["income"] = y_train
    ds_rw = DataFrame(train_rw); ds_rw.targets, ds_rw.sensitive = "income", "sex"
    rw_proc = Reweighing(); ds_rw_t = rw_proc.fit_transform(ds_rw)
    rw_clf = train_classifier(X_train, y_train, sample_weight=ds_rw_t["weights"].values)
    rw_pred = rw_clf.predict(X_test)
    rw_acc = accuracy_score(y_test, rw_pred)
    rw_spd, rw_di = evaluate_fairness(X_test, rw_pred)
    results.append({"algorithm": "Reweighing", "accuracy": rw_acc, "spd": rw_spd, "di": rw_di})

    # ---------------------- Disparate Impact Remover ----------------------
    train_dir = X_train.copy(); train_dir["income"] = y_train
    ds_dir = DataFrame(train_dir); ds_dir.targets, ds_dir.sensitive = "income", "sex"
    dir_proc = DisparateImpactRemover(repair_level=1.0)
    train_dir_t = dir_proc.fit_transform(ds_dir).drop(columns=["sex"])
    dir_clf = train_classifier(train_dir_t, y_train)
    dir_pred = dir_clf.predict(X_test.drop(columns=["sex"]))
    dir_acc = accuracy_score(y_test, dir_pred)
    dir_spd, dir_di = evaluate_fairness(X_test, dir_pred)
    results.append({"algorithm": "DIR", "accuracy": dir_acc, "spd": dir_spd, "di": dir_di})

    # -------------------- Learning Fair Representations -------------------
    # Per LFR servono input_dim/latent_dim/output_dim quando non si passano reti custom
    latent_dim = 8  # scelta ragionevole
    lfr_proc = LFR(input_dim=X_train.shape[1], latent_dim=latent_dim, output_dim=X_train.shape[1])

    # Preparazione dati train per LFR
    train_lfr_df = X_train.copy(); train_lfr_df["income"] = y_train
    ds_lfr_train = DataFrame(train_lfr_df); ds_lfr_train.targets, ds_lfr_train.sensitive = "income", "sex"

    ds_lfr_latent = lfr_proc.fit_transform(ds_lfr_train, epochs=60, learning_rate=0.001)
    X_train_lfr = pd.DataFrame(ds_lfr_latent.values, columns=ds_lfr_latent.columns)

    lfr_clf = train_classifier(X_train_lfr, y_train)

    # Trasformazione del test
    test_lfr_df = X_test.copy(); test_lfr_df["income"] = y_test
    ds_lfr_test = DataFrame(test_lfr_df); ds_lfr_test.targets, ds_lfr_test.sensitive = "income", "sex"
    X_test_lfr_df = lfr_proc.transform(ds_lfr_test)
    X_test_lfr = pd.DataFrame(X_test_lfr_df.values, columns=X_test_lfr_df.columns)

    lfr_pred = lfr_clf.predict(X_test_lfr)
    lfr_acc = accuracy_score(y_test, lfr_pred)
    lfr_spd, lfr_di = evaluate_fairness(X_test, lfr_pred)

    results.append({"algorithm": "LFR", "accuracy": lfr_acc, "spd": lfr_spd, "di": lfr_di})

    # --------------------------- Output risultati --------------------------
    print("\n----- RISULTATI -----")
    for r in results:
        print(f"{r['algorithm']:<10} Acc: {r['accuracy']:.4f}  SPD: {r['spd']:.4f}  DI: {r['di']:.4f}")

    # Salva grafici
    save_plots(results)
    return results


if __name__ == "__main__":
    main()
