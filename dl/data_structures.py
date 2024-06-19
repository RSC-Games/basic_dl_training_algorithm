import numpy as np

# Dataset representation.
class DataSet:
    def __init__(self, set_input: np.ndarray, set_output: np.ndarray):
        self.X = set_input
        self.Y = set_output


# Layer output info from the linear forward step.
class LinearOutput:
    def __init__(self, Z: np.ndarray, A: np.ndarray, W: np.ndarray, b: np.ndarray):
        self.Z = Z
        self.cache_A = A
        self.cache_W = W
        self.cache_b = b


# Contains important activation data.
class ActivationStruct:
    def __init__(self, A: np.ndarray, Z: np.ndarray):
        self.A = A
        self.Z = Z

    def add_linear(self, linear: LinearOutput):
        self.linear_cache = linear


# Contains derivatives of essential activation data.
class ActivationDerivatives:
    def __init__(self, dA_prev: np.ndarray, dW: np.ndarray, db: np.ndarray):
        self.dA_prev = dA_prev
        self.dW = dW
        self.db = db


# Contains all of the activation data for the entire network.
class NetworkActivation:
    def __init__(self, layers: list[ActivationStruct]):
        self.layer_activation = layers


# Contains individual layer data.
class NetworkLayer:
    def __init__(self, n_nodes: int, n_prev: int):
        self.W = np.random.randn(n_nodes, n_prev) / np.sqrt(n_prev) # or use * 0.01?
        self.b = np.zeros((n_nodes, 1))