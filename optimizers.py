import numpy as np


class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def __call__(self, param: np.ndarray, gradient: np.ndarray):
        new_param = param - self.learning_rate * gradient
        return new_param
