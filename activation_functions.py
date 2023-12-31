import numpy as np

MAX_EXP_INPUT = 50
MIN_EXP_INPUT = -50


class Linear:
    def __init__(self):
        pass

    def __call__(self, z: np.ndarray):
        a = z
        return a

    @staticmethod
    def differentiate(z: np.ndarray):
        d = np.ones(shape=z.shape[0])
        return d


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, z: np.ndarray):
        z = np.where(z >= MAX_EXP_INPUT, MAX_EXP_INPUT, z)
        z = np.where(z <= MIN_EXP_INPUT, MIN_EXP_INPUT, z)
        a = 1 / (1 + np.exp(-z))
        return a

    @staticmethod
    def differentiate(z: np.ndarray):
        z = np.where(z >= MAX_EXP_INPUT, MAX_EXP_INPUT, z)
        z = np.where(z <= MIN_EXP_INPUT, MIN_EXP_INPUT, z)
        d1 = np.exp(-z)
        d2 = np.multiply(1 + np.exp(-z), 1 + np.exp(-z))
        d = d1 / d2
        # d = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        return d


class Relu:
    def __init__(self):
        pass

    def __call__(self, z: np.ndarray):
        a = np.where(z > 0, z, 0)
        return a

    @staticmethod
    def differentiate(z: np.ndarray):
        # Derivative of the ReLU is undefined at z = 0, but it is assumed to have derivative 0
        d = np.where(z > 0, 1, 0)
        return d
