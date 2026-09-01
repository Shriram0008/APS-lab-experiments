from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class ModelTraining:

    def create_model(self):

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000
            )
        )

        return model


    def train_model(
        self,
        model,
        X_train,
        y_train
    ):

        model.fit(
            X_train,
            y_train
        )

        print(
            "\nModel trained successfully."
        )

        return model