from fairlib import DataFrame
import numpy.random as random

COLUMNS = ["height", "sex", "class"]
NEW_COLUMNS = ["age", "weight", "income"]
ALL_COLUMNS = COLUMNS + NEW_COLUMNS

RATIO_FEMALE = 0.3
THRESHOLD_TALL_MALE = 170
THRESHOLD_TALL_FEAMLE = 150
DEFAULT_SIZE = 10000


def biased_dataset_people_height(size=DEFAULT_SIZE, binary=False):
    """
    Generates a biased dataset simulating a population with height, sex, and additional attributes.

    This function creates a dataset with a specified number of individuals, assigning them:
    - Height based on a normal distribution.
    - Sex with a predefined probability bias.
    - A class label ('tall' or 'short') depending on a height threshold.
    - Additional attributes like age, weight, and income with specific distributions.

    If `binary` is set to True, the function converts the class labels into binary values and
    adds a 'male' column replacing the 'sex' column.

    Parameters:
    size (int): Number of samples to generate. Default is 10,000.
    binary (bool): Whether to convert class labels to binary and replace 'sex' with 'male'. Default is False.

    Returns:
    DataFrame: A dataset with the generated attributes.
    """
    data = []
    for i in range(size):
        # Selecting sex with bias
        sex = "F" if random.uniform(0, 1) < RATIO_FEMALE else "M"
        # Height generated from a normal distribution
        height = random.normal((THRESHOLD_TALL_FEAMLE + THRESHOLD_TALL_MALE) / 2, 25.0)
        threshold = THRESHOLD_TALL_FEAMLE if sex == "F" else THRESHOLD_TALL_MALE
        klass = "tall" if height >= threshold else "short"

        # Adding additional characteristics
        # Age: uniformly distributed between 18 and 65 years
        age = random.randint(18, 66)  # 66 is excluded, so age 18-65

        # Weight: on average lower for women and slightly higher for men
        if sex == "F":
            weight = random.normal(65, 10)
        else:
            weight = random.normal(75, 10)

        # Income: generated with different mean and variance based on sex
        if sex == "F":
            income = random.normal(3000, 500)
        else:
            income = random.normal(3500, 500)

        data.append([height, sex, klass, age, weight, income])

    df = DataFrame(data, columns=ALL_COLUMNS)

    if binary:
        # Convert the 'class' column to binary and create the 'male' column
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
