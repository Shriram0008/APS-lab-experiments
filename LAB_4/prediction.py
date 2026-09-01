import pandas as pd


class ModelPrediction:

    def get_probabilities(
        self,
        model,
        X_test
    ):

        probabilities = (
            model.predict_proba(X_test)
        )

        print(
            "\nPredicted Probabilities"
        )

        print(
            probabilities[:5]
        )

        return probabilities


    def check_probability_sum(
        self,
        probabilities
    ):

        print(
            "\nProbability Sum"
        )

        print(
            probabilities[:5]
            .sum(axis=1)
        )


    def create_results_table(
        self,
        y_test,
        probabilities
    ):

        # Same as notebook
        results = pd.DataFrame({
            "Actual_class":
                y_test.values,

            "P_malignant":
                probabilities[:, 0],

            "P_benign":
                probabilities[:, 1]
        })

        print(
            "\nProbability Results"
        )

        print(
            results.head(10)
        )

        return results


    def add_actual_labels(
        self,
        results
    ):

        # Same mapping used in notebook
        results["Actual_label"] = (
            results["Actual_class"].map({
                0: "Malignant",
                1: "Benign"
            })
        )

        print(
            "\nActual Labels"
        )

        print(
            results[
                [
                    "Actual_label",
                    "P_malignant",
                    "P_benign"
                ]
            ].head(10)
        )

        return results


    def apply_threshold(
        self,
        results,
        threshold=0.50
    ):

        # Same as notebook
        results["Predicted_malignant"] = (
            results["P_malignant"] >= threshold
        ).astype(int)

        results["Predicted_label"] = (
            results[
                "Predicted_malignant"
            ].map({
                1: "Malignant",
                0: "Benign"
            })
        )

        print(
            "\nThreshold:",
            threshold
        )

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


    def compare_thresholds(
        self,
        results
    ):

        print(
            "\nThreshold Comparison"
        )

        for threshold in [
            0.30,
            0.50,
            0.70
        ]:

            predictions = (
                results["P_malignant"]
                >= threshold
            ).astype(int)

            print(
                f"Threshold: {threshold}: "
                f"Predicted malignant cases = "
                f"{predictions.sum()}"
            )