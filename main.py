from layers import Dense
import numpy as np
import activation_functions as af
import neural_network as nn
from losses import BinaryCrossEntropy

x = np.array([[0, 2]])
y = np.array([1])

model = nn.Sequential(
    [
        Dense(units=100, activation=af.Sigmoid()),
        Dense(units=10, activation=af.Sigmoid()),
        Dense(units=1, activation=af.Sigmoid())
    ]
)

# print(model(x))

model.compile(loss=BinaryCrossEntropy())
model.fit(x, y)

# print(model.number_of_params())
