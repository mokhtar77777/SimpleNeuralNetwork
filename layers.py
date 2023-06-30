import numpy as np
import activation_functions as af


class Dense:
    def __init__(self, units, activation=af.Linear()):
        self.weights = None
        self.bias = np.random.rand(1, units) * 2 - 1

        self.units = units
        self.activation = activation

    def __call__(self, a: np.ndarray):
        size_of_input = a.shape[1]

        if self.weights is None:
            self.weights = np.random.rand(size_of_input, self.units) * 2 - 1

        z = self._z(a)
        a = self._forward_prop(z)

        return a

    def _z(self, a: np.ndarray):
        z = np.matmul(a, self.weights) + self.bias
        return z

    def _forward_prop(self, z: np.ndarray):
        a = self.activation(z)
        return a

    def get_weights(self):
        return self.weights, self.bias

    def set_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
