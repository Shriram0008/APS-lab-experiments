from data import generate_linear_data
from visual import plot_2d_data

def main():
    X, y = generate_linear_data()
    plot_2d_data(X, y, title='Linearly Separable Data')

if __name__ == "__main__":
    main()