import numpy as np

TENDS_TO_ZERO = 0.0001
TENDS_TO_ONE = 0.9999


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, a: np.ndarray, y: np.ndarray):
        loss_factor = np.multiply(a - y, a - y)
        mse = np.sum(0.5 * loss_factor)
        return mse

    @staticmethod
    def differentiate(a: np.ndarray, y: np.ndarray):
        d = a - y
        return d


class BinaryCrossEntropy:
    def __init__(self):
        pass

    def __call__(self, a: np.ndarray, y: np.ndarray):
        a = np.where(a == 0, TENDS_TO_ZERO, a)
        a = np.where(a == 1, TENDS_TO_ONE, a)

        bce = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))
        return bce

    @staticmethod
    def differentiate(a: np.ndarray, y: np.ndarray):
        a = np.where(a == 0, TENDS_TO_ZERO, a)
        a = np.where(a == 1, TENDS_TO_ONE, a)

        d = -y * (1 / a) + (1 - y) * (1 / (1 - a))
        return d
