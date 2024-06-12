import numpy as np

# Contains important activation data
class ActivationStruct:
    def __init__(self, A: np.ndarray, Z: np.ndarray):
        self.A = A
        self.Z = Z


# Contains the Neural Network, layers, and all nodes.
class NeuralNetwork:
    def __init__(self, layer_dims: list[int]):
        self.layers = []

        for i in range(1, len(layer_dims)):
            self.layers.append(NetworkLayer(layer_dims[i], layer_dims[i - 1]))


# Contains individual layer data.
class NetworkLayer:
    def __init__(self, n_nodes: int, n_prev: int):
        self.W = np.random.randn(n_nodes, n_prev) / np.sqrt(n_prev) # or use * 0.01?
        self.b = np.zeros((n_nodes, 1))