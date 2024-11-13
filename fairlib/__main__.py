from typing import Iterable

import openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from fairlib.inprocessing import Fauci
from keras import Model, Input, Sequential
from keras.src.layers import Dense

import fairlib as fl
import fairlib.keras as keras

df = fl.DataFrame(
    {
        "name": ["Alice", "Bob", "Carla", "Davide", "Elena"],
        "age": [25, 32, 45, 29, 34],
        "sex": ["F", "M", "F", "M", "F"],
        "income": [40000, 50000, 45000, 53000, 43000],
    }
)

df.targets = "income"
print(df.targets)  # {'income'}

df.sensitive = {"age", "sex"}
print(df.sensitive)  # {'age', 'sex'}

protected = {
    "age": lambda x: x > 30,
    "sex": lambda x: x == "M",
    "income": lambda x: x > 45000,
}

try:
    df.sensitive = {"age", "sex", "missing"}
except Exception as e:
    print(e)  # Column missing not found

df2 = df.drop(
    ["name"], axis=1
)  # random operation creating another DataFrame. are attributes preserved? yes
print(df2.targets)  # {'income'}
print(df2.sensitive)  # {'age', 'sex'}

df3 = df.drop(
    ["sex"], axis=1
)  # random operation creating another DataFrame. what if the operation changes the columns?
print(df3.targets)  # {'income'}
print(df3.sensitive)  # {'age'}

print(
    df.domains
)  # {name: [Alice, Bob, Carla, Davide, Elena]; age: [25, 29, 32, 34, 45]; sex: [F, M]; income: ['40000', '50000', '45000', '53000', '43000']

# Pre-processing
for column, rule in protected.items():
    df[column] = df[column].apply(rule).astype(int)

print(df.domains)
# Metrics
spd = df.statistical_parity_difference()
print(type(spd))

# df.sensitive = {"sex"}

# print(df.domains)
# Apply reweighing for 1 sensitive field
# df_transformed = df.reweighing()
# print(df_transformed)

# Apply reweighing for 2 or more sensitive field
# df.sensitive = {"sex", "age"}
# df_transformed = df.reweighing()
# print(df_transformed)
