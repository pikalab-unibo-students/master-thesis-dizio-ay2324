import unittest
import openml
import fairlib as fl
from fairlib import keras
from fairlib.inprocessing import Fauci
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input


# Function to create a basic neural network model
def create_model(input_shape):
    return Sequential([
        Input((input_shape,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])


# Function to evaluate a model's accuracy and fairness metric
def evaluate_model(model, X_test, y_test, typeOfEval):
    predictions = model.predict(X_test.astype(float))
    y_pred = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Convert test set to FairLib DataFrame
    X_test["income"] = y_pred
    dataset = fl.DataFrame(X_test)
    dataset.targets = "income"
    dataset.sensitive = "sex"

    # Evaluate fairness metric
    if typeOfEval == "spd":
        metric = dataset.statistical_parity_difference()
    elif typeOfEval == "di":
        metric = dataset.disparate_impact()
    else:
        raise ValueError("Metric Not Found")
    return accuracy, metric


# Test class for Fauci fairness-aware learning model
class TestFauci(unittest.TestCase):

    EPOCHS = 10
    BATCH_SIZE = 100
    VALIDATION_SPLIT = 0.33

    def setUp(self):
        # Set random seed for reproducibility
        keras.utils.set_random_seed(420)

        # Load dataset from OpenML
        dataset = openml.datasets.get_dataset(179)
        X, y, _, names = dataset.get_data(target=dataset.default_target_attribute)

        # Impute missing values
        imputer = SimpleImputer(strategy="most_frequent")
        X_imputed = imputer.fit_transform(X)

        # Encode categorical features
        X_discretized = X_imputed.copy()
        for col in X.columns:
            if X[col].dtype == "category":
                le = LabelEncoder()
                X_discretized[:, X.columns.get_loc(col)] = le.fit_transform(
                    X_discretized[:, X.columns.get_loc(col)]
                )

        self.X = fl.DataFrame(X_discretized, columns=names)
        self.num_features = self.X.shape[1]
        self.y = y.apply(lambda x: x == ">50K").astype(int)

    def train_and_evaluate(self, regularizer, fairness_metric):
        # Prepare data for Fauci
        X_train = self.X.copy()
        y_train = self.y.copy()
        X_train["income"] = y_train
        fauci_train_dataset = fl.DataFrame(X_train)
        fauci_train_dataset.targets = "income"
        fauci_train_dataset.sensitive = "sex"
        X_train.drop(columns=["income"], inplace=True)

        # Train Fauci model
        fauci_model = create_model(self.num_features)
        fauciModel = Fauci(
            fauci_model,
            loss="binary_crossentropy",
            regularizer=regularizer,
            optimizer="adam",
            metrics=["accuracy"]
        )
        fauciModel.fit(
            fauci_train_dataset,
            converting_to_type=float,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_split=self.VALIDATION_SPLIT
        )

        # Evaluate Fauci model
        fauci_accuracy, fauci_metric = evaluate_model(
            fauciModel, X_train, y_train, fairness_metric
        )

        # Train default model
        default_model = create_model(self.num_features)
        default_model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        default_model.fit(
            X_train.astype(float),
            y_train.astype(float),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_split=self.VALIDATION_SPLIT
        )

        # Evaluate default model
        default_accuracy, default_metric = evaluate_model(
            default_model, X_train, y_train, fairness_metric
        )

        return fauci_accuracy, fauci_metric, default_accuracy, default_metric

    def testFauciOneSensitiveAttrSPD(self):
        # Test Fauci using Statistical Parity Difference (SPD)
        fauci_accuracy, fauci_spd, default_accuracy, default_spd = self.train_and_evaluate(
            regularizer="sp", fairness_metric="spd"
        )

        # Print results
        print("FAUCI: accuracy:", fauci_accuracy, "spd:", fauci_spd)
        print("Default Model: accuracy:", default_accuracy, "spd:", default_spd)

        # Assert fairness improvement
        assert (
                fauci_spd[{"income": 1, "sex": 1}]
                <= default_spd[{"income": 1, "sex": 1}]
        ), f"Expected {fauci_spd}, to be less than {default_spd}"

    def testFauciOneSensitiveAttrDI(self):
        # Test Fauci using Disparate Impact (DI)
        fauci_accuracy, fauci_di, default_accuracy, default_di = self.train_and_evaluate(
            regularizer="di", fairness_metric="di"
        )

        # Print results
        print("FAUCI: accuracy:", fauci_accuracy, "di:", fauci_di)
        print("Default Model: accuracy:", default_accuracy, "di:", default_di)

        # Assert fairness improvement
        fauci_distance = abs(fauci_di[{"income": 1, "sex": 1}] - 1)
        model_distance = abs(default_di[{"income": 1, "sex": 1}] - 1)
        assert (
                fauci_distance <= model_distance
        ), f"Expected {fauci_distance}, to be less than {model_distance}"


if __name__ == "__main__":
    unittest.main()
