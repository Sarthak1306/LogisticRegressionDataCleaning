import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
# print(data.head())
# print(data.info())
# print(data.describe())

# sns.heatmap(data.isnull())
# plt.show()

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)


data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})


print(data.head())
