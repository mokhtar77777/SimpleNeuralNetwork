import numpy as np
from stack import Stack


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.num_of_layers = len(layers)
        self.loss = None

    def __call__(self, x: np.ndarray):
        return self.predict(x)

    def _create_stacks_fwd_prop(self, x_example: np.ndarray):
        z_stack = Stack(initial_size=self.num_of_layers)
        a_stack = Stack(initial_size=self.num_of_layers + 1)

        a_stack.push(x_example)

        for layer in self.layers:
            cur_z = layer.z(a_stack.top())
            z_stack.push(cur_z)

            cur_a = layer.forward_prop(cur_z)
            a_stack.push(cur_a)

        return z_stack, a_stack

    def fit(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0]
        training_examples = x.shape[0]

        for example_num in range(training_examples):
            z_stack, a_stack = self._create_stacks_fwd_prop(x[example_num])

            print(a_stack.top())

    def compile(self, loss):
        self.loss = loss

    def predict(self, x: np.ndarray):
        cur_activation = x

        for layer in self.layers:
            cur_activation = layer(cur_activation)

        return cur_activation
