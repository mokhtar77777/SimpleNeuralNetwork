import numpy as np
from stack import Stack


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.num_of_layers = len(layers)
        self.loss = None
        self.optimizer = None

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

    def _update_params_in_layer(self, layer, training_examples):
        weights, bias = layer.get_weights()
        weights_gradient = layer.temp_weights_gradient / training_examples
        bias_gradient = layer.temp_bias_gradient / training_examples

        new_weights = self.optimizer(weights, weights_gradient)
        new_bias = self.optimizer(bias, bias_gradient)

        layer.set_weights(weights=new_weights, bias=new_bias)

    def number_of_params(self):
        params = 0

        shape_of_a = self.input_shape
        for layer in self.layers:
            units = layer.get_num_of_units()
            params = params + units * shape_of_a + units

            shape_of_a = units

        return params

    def calculate_loss(self, x: np.ndarray, y: np.ndarray):
        prediction = self(x)
        loss = self.loss(prediction, y)
        return loss

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=1):
        assert x.shape[0] == y.shape[0]
        assert self.optimizer is not None
        assert self.loss is not None
        self.input_shape = x.shape[1]
        training_examples = x.shape[0]

        for epoch_num in range(epochs):
            for example_num in range(training_examples):
                z_stack, a_stack = self._create_stacks_fwd_prop(x[example_num])

                dj_da = self.loss.differentiate(a_stack.top(), y[example_num])

                for layer in self.layers[::-1]:
                    cur_layer_activation = layer.activation
                    cur_layer_units = layer.get_num_of_units()
                    cur_layer_weights, _ = layer.get_weights()

                    da_dz = cur_layer_activation.differentiate(z_stack.top())
                    z_stack.pop()

                    a_stack.pop()
                    dz_dw = np.tile(a_stack.top(), (cur_layer_units, 1))

                    dj_db = np.multiply(dj_da.T, da_dz.T)
                    dj_dw = np.multiply(dj_db, dz_dw)
                    dj_da = dj_da @ (np.multiply(da_dz.T, cur_layer_weights.T))

                    dj_dw = dj_dw.T
                    dj_db = dj_db.T

                    layer.temp_weights_gradient += dj_dw
                    layer.temp_bias_gradient += dj_db

            for layer in self.layers:
                self._update_params_in_layer(layer, training_examples)

            loss = self.calculate_loss(x, y)

            print(f"Epoch {epoch_num + 1} - >>>>>>>>>> loss = {loss}\n")

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, x: np.ndarray):
        cur_activation = x

        for layer in self.layers:
            cur_activation = layer(cur_activation)

        return cur_activation
