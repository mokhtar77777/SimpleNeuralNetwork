import numpy as np


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.num_of_layers = len(layers)

    def __call__(self, x: np.ndarray):
        return self.predict(x)

    def predict(self, x: np.ndarray):
        cur_activation = x

        for layer in self.layers:
            cur_activation = layer(cur_activation)

        return cur_activation
