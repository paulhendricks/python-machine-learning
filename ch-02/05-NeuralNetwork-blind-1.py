import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
eta = 0.1

synapse_0 = 2 * np.random.random((2, 10)) - 1
synapse_1 = 2 * np.random.random((10, 1)) - 1

for _ in range(10000):
    layer_0 = X
    layer_1 = 1 / (1 + np.exp(-np.dot(layer_0, synapse_0)))
    layer_2 = 1 / (1 + np.exp(-np.dot(layer_1, synapse_1)))
    layer_2_delta = (layer_2 * (1 - layer_2)) * (y - layer_2)
    layer_1_delta = (layer_1 * (1 - layer_1)) * np.dot(layer_2_delta, synapse_1.T)
    synapse_1 += np.dot(layer_1.T, layer_2_delta)
    synapse_0 += np.dot(layer_0.T, layer_1_delta)
