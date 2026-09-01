from data import (
    load_data,
    create_train_test_data,
    get_class_distribution
)

from visual import plot_class_distribution

from task1 import (
    train_model,
    get_probabilities,
    create_probability_results,
    apply_threshold,
    compare_thresholds
)

from task2 import (
    confusion_matrices,
    calculate_specificity,
    calculate_all_metrics,
    sklearn_verification
)


def main():

    print("========================================")
    print("          APS LAB - 4")
    print("   Understanding Probability in")
    print("       Binary Classification")
    print("========================================")

    # ------------------------------------
    # 1. Load Dataset
    # ------------------------------------

    X, y, data = load_data()

    # ------------------------------------
    # 2. Class Distribution
    # ------------------------------------

    class_distribution = get_class_distribution(
        y,
        data
    )

    # ------------------------------------
    # 3. Visualization
    # ------------------------------------

    plot_class_distribution(
        class_distribution
    )

    # ------------------------------------
    # 4. Train/Test Split
    # ------------------------------------

    X_train, X_test, y_train, y_test = (
        create_train_test_data(X, y)
    )

    # ------------------------------------
    # 5. Train Logistic Regression
    # ------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # ------------------------------------
    # 6. Obtain Probabilities
    # ------------------------------------

    probabilities = get_probabilities(
        model,
        X_test
    )

    # ------------------------------------
    # 7. Create Probability Results
    # ------------------------------------

    results = create_probability_results(
        y_test,
        probabilities
    )

    # ------------------------------------
    # 8. Apply Threshold
    # ------------------------------------

    results = apply_threshold(
        results,
        threshold=0.50
    )

    # ------------------------------------
    # 9. Compare Thresholds
    # ------------------------------------

    compare_thresholds(results)

    # ------------------------------------
    # 10. Confusion Matrices
    # ------------------------------------

    confusion_matrices(
        y_test,
        probabilities
    )

    # ------------------------------------
    # 11. Specificity
    # ------------------------------------

    calculate_specificity(
        y_test,
        probabilities
    )

    # ------------------------------------
    # 12. Compare All Metrics
    # ------------------------------------

    calculate_all_metrics(
        y_test,
        probabilities
    )

    # ------------------------------------
    # 13. Verify using Sklearn
    # ------------------------------------

    sklearn_verification(
        y_test,
        results
    )

    print("\n========================================")
    print("          LAB-4 COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()