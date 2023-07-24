import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_clean_data():
    data = pd.read_csv("data.csv")
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def main():
    data = get_clean_data()

    print(data.head())


if __name__ == "__main__":
    main()
