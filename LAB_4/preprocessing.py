import pandas as pd

from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def show_class_distribution(
        self,
        y,
        data
    ):

        class_counts = (
            y.value_counts()
            .sort_index()
        )

        # Keep the notebook's original structure
        class_distribution = pd.DataFrame({
            "Class": data.target_names,
            "Count": class_counts.values,
            "Probability":
                class_counts.values / len(y)
        })

        print("\nClass Distribution")
        print(class_distribution)

        return class_distribution


    def split_data(
        self,
        X,
        y
    ):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        print(
            "\nTraining size:",
            len(y_train)
        )

        print(
            "Testing size:",
            len(y_test)
        )

        print("\nTraining Proportion")
        print(
            y_train
            .value_counts(normalize=True)
            .sort_index()
        )

        print("\nTesting Proportion")
        print(
            y_test
            .value_counts(normalize=True)
            .sort_index()
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )