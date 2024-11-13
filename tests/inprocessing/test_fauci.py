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
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def evaluate_model(model, X_test, y_test, typeOfEval):
    predictions = model.predict(X_test.astype(float))
    y_pred = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    X_test["income"] = y_pred
    dataset = fl.DataFrame(X_test)
    dataset.targets = 'income'
    dataset.sensitive = 'sex'
    if typeOfEval == "spd":
        metric = dataset.statistical_parity_difference()
    elif typeOfEval == "di":
        metric = dataset.disparate_impact()
    else:
        raise "Metric Not Found"
    return accuracy, metric


class TestFauci(unittest.TestCase):

    def setUp(self):
        keras.utils.set_random_seed(423)
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
        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train['income'] = y_train
        fauci_train_dataset = fl.DataFrame(X_train)
        fauci_train_dataset.targets = "income"
        fauci_train_dataset.sensitive = 'sex'
        fauci_model = create_model()
        X_train.drop(columns=['income'], inplace=True)

        fauciModel = Fauci(fauci_model, loss='binary_crossentropy', regularizer='sp', optimizer='adam',
                           metrics=['accuracy'])
        fauciModel.fit(fauci_train_dataset, converting_to_type=float, epochs=10, batch_size=32, validation_split=0.2)
        fauci_accuracy, fauci_spd = evaluate_model(fauciModel, X_train, y_train, "spd")

        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.35, random_state=42)
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        default_model = create_model()
        default_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        default_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        default_model_accuracy, default_model_spd = evaluate_model(default_model, X_train, y_train, "spd")

        print("FAUCI: accuracy: ", fauci_accuracy, " spd: ", fauci_spd)
        print("Default Model: accuracy: ", default_model_accuracy, " spd: ", default_model_spd)

        assert (
                fauci_spd[{'income': 1, 'sex': 1}] <= default_model_spd[{'income': 1, 'sex': 1}]
        ), f"Expected {fauci_spd}, to be less than {default_model_spd}"

    def testFauciOneSensitiveAttrDI(self):
        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.35, random_state=1)
        X_train['income'] = y_train
        fauci_train_dataset = fl.DataFrame(X_train)
        fauci_train_dataset.targets = "income"
        fauci_train_dataset.sensitive = 'sex'
        fauci_model = create_model()
        X_train.drop(columns=['income'], inplace=True)

        fauciModel = Fauci(fauci_model, loss='binary_crossentropy', regularizer='di', optimizer='adam',
                           metrics=['accuracy'])
        fauciModel.fit(fauci_train_dataset, converting_to_type=float, epochs=10, batch_size=32, validation_split=0.2)
        fauci_accuracy, fauci_di = evaluate_model(fauciModel, X_train, y_train, "di")

        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.35, random_state=1)
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        default_model = create_model()
        default_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        default_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        default_model_accuracy, default_model_di = evaluate_model(default_model, X_train, y_train, "di")

        print("FAUCI: accuracy: ", fauci_accuracy, " di: ", fauci_di)
        print("Default Model: accuracy: ", default_model_accuracy, " di: ", default_model_di)

        fauci_distance = abs(fauci_di[{'income': 1, 'sex': 1}] - 1)
        model_distance = abs(default_model_di[{'income': 1, 'sex': 1}] - 1)

        assert (
                fauci_distance <= model_distance
        ), f"Expected {fauci_distance}, to be less than {model_distance}"


if __name__ == "__main__":
    unittest.main()
