import numpy as np
from numba import jit
from activation_functions import relu, sigmoid
from model import sim_layer, calc_cost

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


# Contains the Neural Network, layers, and all nodes.
class NeuralNetwork:
    def __init__(self, layer_dims: list[int]):
        self.layers = []

        for i in range(1, len(layer_dims)):
            self.layers.append(NetworkLayer(layer_dims[i], layer_dims[i - 1]))

    def get_layer_count(self) -> int:
        """
        Supply the layer count of this network to the caller.
        
        :returns int: Layer count of this network.
        """
        return len(self.layers)
    
    def get_layer(self, layer_id: int) -> NetworkLayer:
        """
        Supply the requested layer.
        
        :returns NetworkLayer: The requested layer.
        """
        return self.layers[layer_id]


    @jit()
    def train(self, set: DataSet, learning_rate=0.0075, num_iterations=2500, log=False):
        """
        Train a L-layer neural network.
        
        :param set: Input dataset.
        :param learning_rate: Learning rate of gradient descent.
        :param num_iterations: Iterations for optimization.
        """
        AL = set.X
        print(f"[Info]: Training Nnet model at lr {learning_rate} for {num_iterations} iterations.")        
        
        for i in range(0, num_iterations):
            # Forward propagation.
            AL = self.model_forward(set.X)

            # Calculate cost.
            cost = calc_cost(AL, set.Y)

            # Backward propagation.
            grads = self.model_backward(AL, set.Y)

    
    @jit()
    def model_forward(nnet, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation step for a n-level neural network. This network
        is assumed to use RELU nodes up to the output node, which is sigmoid.
        
        :param X: Input layer
        :param nnet: Network layer weights and biases.
        :returns list: Returns a list of the output activation value and the
            cached activation values.
        """

        caches = []
        A = X
        L = nnet.get_layer_count()

        for layer in range(1, L):
            A_prev = A
            current_layer = nnet.get_layer(layer)
            activation = sim_layer(A_prev, current_layer.W, current_layer.b, relu)
            A = activation.A
            caches.append(activation)

        current_layer = nnet.get_layer(L)
        activation = sim_layer(A, current_layer.W, current_layer.b, sigmoid)
        AL = activation.A
        caches.append(activation)

        # Store the cached activation data for the backprop stage.
        nnet.last_iter_caches = caches

        return AL
    

    @jit()
    def model_backward(self, AL: np.ndarray, Y: np.ndarray):
        pass