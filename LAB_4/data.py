import pandas as pd
from sklearn.datasets import load_breast_cancer


class DataLoader:

    def load_data(self):

        # Load dataset
        data = load_breast_cancer()

        # Feature matrix
        X = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )

        # Target
        # 0 = benign
        # 1 = malignant
        y = pd.Series(
            (data.target == 0).astype(int),
            name="malignant"
        )

        print(y.value_counts())

        print(
            "Feature matrix shape:",
            X.shape
        )

        print(
            "Target shape:",
            y.shape
        )

        print(
            "Class names:",
            data.target_names
        )

        return X, y, data