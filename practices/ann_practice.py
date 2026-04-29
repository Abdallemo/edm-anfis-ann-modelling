import numpy as np
from numpy.typing import NDArray


class NeuralNetwork:
    def __init__(self, *layer_sizes: int) -> None:
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            new_layer = Layer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(new_layer)

    def calculate_outputs(self, inputs: np.ndarray):

        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def fit(self, X_train, y_train):
        pass

    def predict(self, x_test):

        return self.calculate_outputs(x_test)

    def cost(self):

        pass


class Layer:
    def __init__(self, num_nodes_in: int, num_nodes_out: int) -> None:
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.weights = np.zeros((num_nodes_in, num_nodes_out))
        self.biases = np.zeros(num_nodes_out)
        pass

    def calculate_outputs(self, inputs: np.ndarray):
        weighted_sum = inputs @ self.weights + self.biases
        activation = activation_function(weighted_sum)
        return activation


def sigmoid(weighted_input: NDArray[np.float64]):
    return 1 / (1 + np.exp(-weighted_input))


def tanh(weighted_input: NDArray[np.float64]):
    e2w = np.exp(2 * weighted_input)
    return (e2w - 1) / (e2w + 1)


def silu(weighted_input: NDArray[np.float64]):
    return weighted_input / (1 + np.exp(-weighted_input))


def relu(weighted_input: NDArray[np.float64]):
    return np.maximum(0, weighted_input)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "hyperbolic_tangnet": tanh,
    "silu": silu,
    "relu": relu,
}


def activation_function(weighted_input, fn_type="sigmoid"):
    try:
        return ACTIVATIONS[fn_type](weighted_input)
    except KeyError:
        raise ValueError(f"Unknown activation function: {fn_type}")


def evaluate(predictions, y_test):
    return 1, 2, 3


def node_cost(output_activation, expected_output):
    error = output_activation - expected_output
    return error * error
