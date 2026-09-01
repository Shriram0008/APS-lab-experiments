import matplotlib.pyplot as plt


def plot_class_distribution(class_distribution):

    class_distribution.plot(
        x="Class",
        y="Count",
        kind="bar",
        legend=False,
        color=["tomato", "steelblue"]
    )

    plt.ylabel("Number of Observations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()