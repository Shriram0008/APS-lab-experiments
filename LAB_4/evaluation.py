import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:

    def calculate_specificity(
        self,
        y_test,
        predicted_malignant
    ):

        cm = confusion_matrix(
            y_test,
            predicted_malignant
        )

        tn = cm[0, 0]
        fp = cm[0, 1]

        specificity = (
            tn / (tn + fp)
        )

        return specificity


    def calculate_metrics(
        self,
        y_test,
        probabilities,
        threshold
    ):

        # IMPORTANT:
        # Same as notebook
        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            predicted_malignant
        )

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        # Specificity
        specificity = (
            tn / (tn + fp)
        )

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            predicted_malignant
        )

        # Precision
        precision = precision_score(
            y_test,
            predicted_malignant,
            zero_division=0
        )

        # Recall
        recall = recall_score(
            y_test,
            predicted_malignant,
            zero_division=0
        )

        # F1
        f1 = f1_score(
            y_test,
            predicted_malignant,
            zero_division=0
        )

        return {
            "Threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1
        }


    def create_final_table(
        self,
        y_test,
        probabilities
    ):

        metric_results = []

        thresholds = [
            0.10,
            0.30,
            0.50,
            0.70,
            0.90
        ]

        for threshold in thresholds:

            result = self.calculate_metrics(
                y_test,
                probabilities,
                threshold
            )

            metric_results.append(
                result
            )

        final_table = pd.DataFrame(
            metric_results
        )

        return final_table


    def show_final_table(
        self,
        y_test,
        probabilities
    ):

        final_table = (
            self.create_final_table(
                y_test,
                probabilities
            )
        )

        print(
            "\nFinal Threshold Comparison"
        )

        print(
            final_table
            .round(3)
            .to_string(index=True)
        )

        return final_table