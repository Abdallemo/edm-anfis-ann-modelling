import pandas as pd

from ann import NeuralNetwork


def build_dataset1():
    df = pd.read_csv("./datasets/dataset1.csv")
    X = df[["volt", "ip", "ton", "toff"]]
    y = df["ra"]

    final_model = NeuralNetwork(
        hidden_layer_sizes=(5, 3),
        activation="logistic",
        alpha=0.75,
        max_iter=5000,
    )
    final_model.fit(X, y)
    final_model.save("models/dataset1_model.pkl")

    pass


def build_dataset2():
    df = pd.read_csv("./datasets/dataset2.csv")
    X = df[["ton", "duty_cycle", "peak_current", "voltage"]]
    y = df["surface_roughness"]

    final_model = NeuralNetwork(
        hidden_layer_sizes=(5,), activation="tanh", alpha=0.01, max_iter=5000
    )

    final_model.fit(X, y)

    final_model.save("models/dataset2_model.pkl")


def main():
    print("building dataset 1 model")
    build_dataset1()
    print("building dataset 2 model")
    build_dataset2()


if __name__ == "__main__":
    main()
