import unittest
import openml

import fairlib as fl
from fairlib import keras
from fairlib.inprocessing import Fauci
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for classification
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test.astype(float)) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    X_test["income"] = y_pred
    spd_dataset = fl.DataFrame(X_test)
    spd_dataset.targets = "income"
    spd_dataset.sensitive = 'sex'
    spd = spd_dataset.statistical_parity_difference()
    return accuracy, spd


class TestFauci(unittest.TestCase):

    def setUp(self):
        keras.utils.set_random_seed(42)
        dataset = openml.datasets.get_dataset(179)
        X, y, _, names = dataset.get_data(target=dataset.default_target_attribute)
        imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = imputer.fit_transform(X)
        X_discretized = X_imputed.copy()
        for col in X.columns:
            if X[col].dtype == 'category':
                le = LabelEncoder()
                X_discretized[:, X.columns.get_loc(col)] = le.fit_transform(X_discretized[:, X.columns.get_loc(col)])
        self.X = fl.DataFrame(X_discretized, columns=names)
        self.y = y.apply(lambda x: x == ">50K").astype(int)

    def testFauciOneSensitiveAttrSPD(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train['income'] = y_train
        fauci_train_dataset = fl.DataFrame(X_train)
        fauci_train_dataset.targets = "income"
        fauci_train_dataset.sensitive = 'sex'
        fauci_model = create_model()

        fauciModel = Fauci(fauci_model, loss='binary_crossentropy', regularizer='sp', optimizer='adam',
                           metrics=['accuracy'])
        fauciModel.fit(fauci_train_dataset, converting_to_type=float, epochs=5, batch_size=32, validation_split=0.1)
        fauci_accuracy, fauci_spd = evaluate_model(fauciModel, X_test, y_test)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        default_model = create_model()
        default_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        default_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
        default_model_accuracy, default_model_spd = evaluate_model(default_model, X_test, y_test)

        print("FAUCI: accuracy: ", fauci_accuracy, " spd: ", fauci_spd)
        print("Default Model: accuracy: ", default_model_accuracy, " spd: ", default_model_spd)

        assert (
                fauci_spd[{'income': 1, 'sex': 1}] >= default_model_spd[{'income': 1, 'sex': 1}]
        ), f"Expected {fauci_spd}, to be less than {default_model_spd}"

    def testFauciOneSensitiveAttrDI(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train['income'] = y_train
        fauci_train_dataset = fl.DataFrame(X_train)
        fauci_train_dataset.targets = "income"
        fauci_train_dataset.sensitive = 'sex'
        fauci_model = create_model()

        fauciModel = Fauci(fauci_model, loss='binary_crossentropy', regularizer='di', optimizer='adam',
                           metrics=['accuracy'])
        fauciModel.fit(fauci_train_dataset, converting_to_type=float, epochs=5, batch_size=32, validation_split=0.1)
        fauci_accuracy, fauci_di = evaluate_model(fauciModel, X_test, y_test)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        default_model = create_model()
        default_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        default_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
        default_model_accuracy, default_model_di = evaluate_model(default_model, X_test, y_test)

        print("FAUCI: accuracy: ", fauci_accuracy, " spd: ", fauci_di)
        print("Default Model: accuracy: ", default_model_accuracy, " spd: ", default_model_di)

        assert (
                fauci_di[{'income': 1, 'sex': 1}] >= default_model_di[{'income': 1, 'sex': 1}]
        ), f"Expected {fauci_di}, to be less than {default_model_di}"


if __name__ == "__main__":
    unittest.main()
