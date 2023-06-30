from layers import Dense
import numpy as np
import activation_functions as af

x = np.array([[0, 2]])

layer1 = Dense(units=100, activation=af.Relu())
layer2 = Dense(units=10, activation=af.Relu())
layer3 = Dense(units=1, activation=af.Sigmoid())

a1 = layer1(x)
a2 = layer2(a1)
a3 = layer3(a2)

print(a3)
