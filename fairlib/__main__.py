from typing import Iterable
from fairlib.inprocessing import Fauci
from keras import Model, Input
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

df.sensitive = {"sex"}

print(df.domains)
# Apply reweighing for 1 sensitive field
df_transformed = df.reweighing()
print(df_transformed)

# Apply reweighing for 2 or more sensitive field
df.sensitive = {"sex", "age"}
df_transformed = df.reweighing()
print(df_transformed)

# FaUCI example
NEURONS_PER_LAYER = [100, 50]


def create_fully_connected_nn_tf(
        input_size: int,
        output_size: int,
        neurons_per_hidden_layer: Iterable[int],
        activation_function: str = 'relu',
        latest_activation_function: str = 'sigmoid'
) -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)


# make simple model sequential
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(2,)))
# initialize Fauci
lambda_value = 1
protected_type = 1
f = Fauci(model, protected_type, lambda_value)
# fit model

train = fl.DataFrame(
    {
        "age":    [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        "sex":    [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        "income": [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    }
)

train.targets = "income"
train.sensitive = {"sex"}

valid = fl.DataFrame(
    {
        "age":    [0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        "sex":    [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
        "income": [1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    }
)

valid.targets = "income"
valid.sensitive = {"sex"}

f.fit(train, valid, 10, 1)
