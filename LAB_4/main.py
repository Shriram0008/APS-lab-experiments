from data import DataLoader
from preprocessing import DataPreprocessor
from training import ModelTraining
from prediction import ModelPrediction
from evaluation import ModelEvaluation
from visualization import DataVisualization


class APSLab4:

    def run(self):

        print("======================================")
        print("             APS LAB - 4")
        print("       Binary Classification")
        print("======================================")


        # ==================================
        # 1. LOAD DATA
        # ==================================

        data_loader = DataLoader()

        X, y, data = (
            data_loader.load_data()
        )


        # ==================================
        # 2. PREPROCESSING
        # ==================================

        preprocessor = DataPreprocessor()

        class_distribution = (
            preprocessor.show_class_distribution(
                y,
                data
            )
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = preprocessor.split_data(
            X,
            y
        )


        # ==================================
        # 3. VISUALIZATION
        # ==================================

        visualization = DataVisualization()

        visualization.plot_class_distribution(
            class_distribution
        )


        # ==================================
        # 4. MODEL TRAINING
        # ==================================

        training = ModelTraining()

        model = (
            training.create_model()
        )

        model = (
            training.train_model(
                model,
                X_train,
                y_train
            )
        )


        # ==================================
        # 5. PREDICTION
        # ==================================

        prediction = ModelPrediction()

        probabilities = (
            prediction.get_probabilities(
                model,
                X_test
            )
        )

        prediction.check_probability_sum(
            probabilities
        )

        results = (
            prediction.create_results_table(
                y_test,
                probabilities
            )
        )

        results = (
            prediction.add_actual_labels(
                results
            )
        )

        results = (
            prediction.apply_threshold(
                results,
                threshold=0.50
            )
        )

        prediction.compare_thresholds(
            results
        )


        # ==================================
        # 6. EVALUATION
        # ==================================

        evaluation = ModelEvaluation()

        final_table = (
            evaluation.show_final_table(
                y_test,
                probabilities
            )
        )


        # ==================================
        # COMPLETE
        # ==================================

        print(
            "\n======================================"
        )

        print(
            "           LAB-4 COMPLETED"
        )

        print(
            "======================================"
        )


if __name__ == "__main__":

    lab = APSLab4()

    lab.run()