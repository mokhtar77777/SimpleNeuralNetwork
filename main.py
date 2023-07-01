from layers import Dense
import numpy as np
import activation_functions as af
import neural_network as nn
from optimizers import GradientDescent
from losses import BinaryCrossEntropy, MeanSquaredError


x = np.array(
    [
        [1, 2],
        [0, 0]
    ]
)

y = np.array(
    [[1], [0]]
)

model = nn.Sequential(
    [
        Dense(units=3, activation=af.Relu()),
        Dense(units=1, activation=af.Sigmoid())
    ]
)

model.compile(loss=BinaryCrossEntropy(),
              optimizer=GradientDescent(learning_rate=0.1))

inference_before_fitting = model(x)
before_fitting = np.where(inference_before_fitting >= 0.5, 1, 0)

model.fit(x, y, epochs=500)


inference_after_fitting = model.predict(x)
after_fitting = np.where(inference_after_fitting >= 0.5, 1, 0)
print(f"Inference before fitting:")
print(before_fitting)
print(f"Inference after fitting")
print(after_fitting)
