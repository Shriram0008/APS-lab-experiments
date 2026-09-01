# This handles:

# Confusion Matrix
# Accuracy
# Precision
# Recall
# F1
# Specificity
# Comparison of all thresholds

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def confusion_matrices(y_test, probabilities):

    print("\n--- Confusion Matrices ---")

    actual_malignant = (
        y_test.values == 0
    ).astype(int)

    for threshold in [0.30, 0.50, 0.70]:

        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant,
            predicted_malignant
        )

        print(f"\nThreshold = {threshold}")
        print(cm)


def calculate_specificity(y_test, probabilities):

    print("\n--- Specificity ---")

    for threshold in [0.10, 0.30, 0.50, 0.70, 0.90]:

        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predicted_malignant
        )

        tn = cm[0, 0]
        fp = cm[0, 1]

        specificity = tn / (tn + fp)

        print(
            f"Threshold={threshold}: "
            f"Specificity={specificity}"
        )


def calculate_all_metrics(y_test, probabilities):

    print("\n--- All Metrics Across Thresholds ---")

    metric_results = []

    for threshold in [0.10, 0.30, 0.50, 0.70, 0.90]:

        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predicted_malignant
        )

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        specificity = tn / (tn + fp)

        metric_results.append({
            "Threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,

            "Accuracy": accuracy_score(
                y_test,
                predicted_malignant
            ),

            "Precision": precision_score(
                y_test,
                predicted_malignant,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predicted_malignant,
                zero_division=0
            ),

            "Specificity": specificity,

            "F1": f1_score(
                y_test,
                predicted_malignant,
                zero_division=0
            )
        })

    result_df = pd.DataFrame(metric_results)

    print(result_df.round(3))

    return result_df


def sklearn_verification(y_test, results):

    print("\n--- Sklearn Verification ---")

    y_pred = results["Predicted_malignant"]

    print(
        "Sklearn accuracy:",
        accuracy_score(y_test, y_pred)
    )

    print(
        "Sklearn precision:",
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print(
        "Sklearn recall:",
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print(
        "Sklearn f1:",
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )