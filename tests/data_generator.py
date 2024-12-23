from fairlib import DataFrame
import numpy.random as random


COLUMNS = ["height", "sex", "class"]
RATIO_FEMALE = 0.3
THRESHOLD_TALL_MALE = 170
THRESHOLD_TALL_FEAMLE = 150
DEFAULT_SIZE = 10000


def biased_dataset_people_height(size=DEFAULT_SIZE, binary=False):
    data = []
    for i in range(size):
        sex = "F" if random.uniform(0, 1) < RATIO_FEMALE else "M"
        height = random.normal(
            abs(THRESHOLD_TALL_FEAMLE + THRESHOLD_TALL_MALE) / 2, 25.0
        )
        threshold = THRESHOLD_TALL_FEAMLE if sex == "F" else THRESHOLD_TALL_MALE
        klass = "tall" if height >= threshold else "short"
        data.append([height, sex, klass])
    df = DataFrame(data, columns=COLUMNS)
    if binary:
        df["class"] = df["class"].apply(lambda x: 1 if x == "tall" else 0)
        df["male"] = df["sex"].apply(lambda x: 1 if x == "M" else 0)
        df.drop(columns=["sex"], inplace=True, axis=1)
    df.targets = "class"
    df.sensitive = "male" if binary else "sex"
    return df


if __name__ == "__main__":
    df = biased_dataset_people_height()
    print("#", *df.columns, sep=", ")
    for i in range(len(df)):
        print(*df.loc[i], sep=", ")
    print("# SP", df.statistical_parity_difference())
    print("# DI", df.disparate_impact())
