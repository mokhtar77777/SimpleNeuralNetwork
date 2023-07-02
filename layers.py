import numpy as np
import activation_functions as af


class Dense:
    def __init__(self, units, activation=af.Linear()):
        self.weights = None
        self.bias = np.random.rand(1, units)

        self.temp_weights_gradient = None
        self.temp_bias_gradient = np.zeros(shape=(1, units))

        self.units = units
        self.activation = activation

    def __call__(self, a: np.ndarray):
        self._adjust_weights(a)

        z = self.z(a)
        a = self.forward_prop(z)

        return a

    def _adjust_weights(self, a: np.ndarray):
        size_of_input = a.shape[1]

        if self.weights is None:
            self.weights = np.random.rand(size_of_input, self.units)

        if self.temp_weights_gradient is None:
            self.temp_weights_gradient = np.zeros(shape=(size_of_input, self.units))

    def z(self, a: np.ndarray):
        self._adjust_weights(a)

        z = np.matmul(a, self.weights) + self.bias
        return z

    def forward_prop(self, z: np.ndarray):
        a = self.activation(z)
        return a

    def get_weights(self):
        return self.weights, self.bias

    def set_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias

    def get_num_of_units(self):
        return self.units
