import numpy as np
from stack import Stack


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.num_of_layers = len(layers)
        self.loss = None

        self.input_shape = None

    def __call__(self, x: np.ndarray):
        return self.predict(x)

    def _create_stacks_fwd_prop(self, x_example: np.ndarray):
        x_example = np.asmatrix(x_example)

        z_stack = Stack(initial_size=self.num_of_layers)
        a_stack = Stack(initial_size=self.num_of_layers + 1)

        a_stack.push(x_example)

        for layer in self.layers:
            cur_z = layer.z(a_stack.top())
            z_stack.push(cur_z)

            cur_a = layer.forward_prop(cur_z)
            a_stack.push(cur_a)

        return z_stack, a_stack

    def number_of_params(self):
        params = 0

        shape_of_a = self.input_shape
        for layer in self.layers:
            units = layer.get_num_of_units()
            params = params + units * shape_of_a + units

            shape_of_a = units

        return params

    def fit(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0]
        self.input_shape = x.shape[1]
        training_examples = x.shape[0]

        params_num = self.number_of_params()

        gradients = np.ndarray(shape=(training_examples, params_num))

        for example_num in range(training_examples):
            z_stack, a_stack = self._create_stacks_fwd_prop(x[example_num])

            self.loss.differentiate(a_stack.top(), y[example_num])

            for layer in self.layers[::-1]:
                da_dz = layer.activation.differentiate(z_stack.top())
                z_stack.pop()
                print(da_dz, "\n")

    def compile(self, loss):
        self.loss = loss

    def predict(self, x: np.ndarray):
        cur_activation = x

        for layer in self.layers:
            cur_activation = layer(cur_activation)

        return cur_activation
