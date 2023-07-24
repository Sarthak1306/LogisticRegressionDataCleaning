import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def create_model(data):
    y = data["diagnosis"]  # target variable
    X = data.drop(["diagnosis"], axis=1)

    # Normalize the data using scikit scaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data for Train,Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model


def get_clean_data():
    data = pd.read_csv("data.csv")
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def main():
    data = get_clean_data()

    scaler, model = create_model(data)
    print(data.head())


if __name__ == "__main__":
    main()
