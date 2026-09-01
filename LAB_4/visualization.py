import matplotlib.pyplot as plt


class DataVisualization:

    def plot_class_distribution(
        self,
        class_distribution
    ):

        class_distribution.plot(
            x="Class",
            y="Count",
            kind="bar",
            legend=False,
            color=[
                "tomato",
                "steelblue"
            ]
        )

        plt.ylabel(
            "Number of Observations"
        )

        plt.title(
            "Class Distribution"
        )

        plt.xticks(
            rotation=0
        )

        plt.show()