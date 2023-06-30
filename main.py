from layers import Dense
import numpy as np
import activation_functions as af
import neural_network as nn
from losses import BinaryCrossEntropy, MeanSquaredError

x = np.array([[11]])
y = np.array([1])

layer1 = Dense(units=3, activation=af.Relu())
# layer2 = Dense(units=2, activation=af.Relu())
layer3 = Dense(units=1, activation=af.Sigmoid())

model = nn.Sequential(
    [
        # layer1,
        layer3
    ]
)

print(model(x), "\n\n\n")

model.compile(loss=BinaryCrossEntropy())
model.fit(x, y)

print("\n\n\n")

# print(model.number_of_params())
