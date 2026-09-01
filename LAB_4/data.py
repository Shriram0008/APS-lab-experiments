import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():

    # Load the dataset
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # 1 = Malignant, 0 = Benign
    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )

    print("\n--- Dataset Information ---")
    print(y.value_counts())

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)

    return X, y, data


def create_train_test_data(X, y):

    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n--- Train/Test Split ---")
    print("Training size:", len(y_train))
    print("Testing size:", len(y_test))

    print("\nTraining Proportion")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nTesting Proportion")
    print(y_test.value_counts(normalize=True).sort_index())

    return X_train, X_test, y_train, y_test


def get_class_distribution(y, data):

    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": data.target_names,
        "Count": class_counts.values,
        "Probability": class_counts.values / len(y)
    })

    print("\n--- Class Distribution ---")
    print(class_distribution)

    return class_distribution