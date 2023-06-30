from layers import Dense
import numpy as np
import activation_functions as af
import neural_network as nn

x = np.array([[0, 2]])

model = nn.Sequential(
    [
        Dense(units=100, activation=af.Relu()),
        Dense(units=10, activation=af.Relu()),
        Dense(units=1, activation=af.Sigmoid())
    ]
)

print(model(x))

model.fit(x, None)

