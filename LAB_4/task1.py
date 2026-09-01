#This handles:
# Logistic Regression
# Training
# Predicted probabilities
# Probability table
# Actual labels
# Threshold predictions
# Comparison of thresholds

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):

    print("\n--- Training Logistic Regression Model ---")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    print("Model trained successfully.")

    return model


def get_probabilities(model, X_test):

    print("\n--- Predicted Probabilities ---")

    probabilities = model.predict_proba(X_test)

    print(probabilities[:5])

    print("\nSum of probabilities:")
    print(probabilities[:5].sum(axis=1))

    return probabilities


def create_probability_results(y_test, probabilities):

    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    results["Actual_label"] = results["Actual_class"].map({
        0: "Malignant",
        1: "Benign"
    })

    print("\n--- Probability Results ---")
    print(
        results[
            ["Actual_label", "P_malignant", "P_benign"]
        ].head(10)
    )

    return results


def apply_threshold(results, threshold=0.50):

    results["Predicted_malignant"] = (
        results["P_malignant"] >= threshold
    ).astype(int)

    results["Predicted_label"] = (
        results["Predicted_malignant"]
        .map({
            1: "Malignant",
            0: "Benign"
        })
    )

    print("\n--- Threshold = 0.50 ---")

    print(
        results[
            [
                "Actual_label",
                "P_malignant",
                "Predicted_label"
            ]
        ].head(10)
    )

    return results


def compare_thresholds(results):

    print("\n--- Comparing Thresholds ---")

    for threshold in [0.30, 0.50, 0.70]:

        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold: {threshold}: "
            f"Predicted malignant cases = {predictions.sum()}"
        )