from typing import Callable
import numpy as np
from numba import jit, njit

from dl.data_structures import ActivationStruct, LinearOutput, ActivationDerivatives, NetworkLayer, DataSet
from dl.backprop import activation_func_lut


# Contains the Neural Network, layers, and all nodes.
class NeuralNetwork:
    def __init__(self, layer_dims: list[tuple[int, Callable]]):
        #np.random.seed(1) # For testing
        self.layers = []

        for i in range(1, len(layer_dims)):
            layer_params = layer_dims[i]
            last_layer = layer_dims[i - 1][0]
            self.layers.append(NetworkLayer(layer_params[0], last_layer, layer_params[1]))

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
    

    def predict(self, ds: DataSet) -> np.ndarray:
        """
        Allow the model to predict results.

        :param X: Input data to predict.
        :param Y: Actual output (for accuracy testing).
        :return ndarray: Output prediction values.
        """

        X = ds.X
        Y = ds.Y

        m = X.shape[1]
        n = self.get_layer_count()
        p = np.zeros((1, m))

        # One stage of forward propagation.
        p_hat = self.model_forward(X)

        # Convert probabilities to binary predictions.
        for i in range(0, p_hat.shape[1]):
            p[0, i] = 1 if p_hat[0, i] > 0.5 else 0

        print(f"[Info]: Accuracy on set: {np.sum((p == Y) / m)}%")
        return p


    #@jit(forceobj=True)
    def train(self, set: DataSet, learning_rate=0.0075, num_iterations=2500, log=False):
        """
        Train a L-layer neural network.
        
        :param set: Input dataset.
        :param learning_rate: Learning rate of gradient descent.
        :param num_iterations: Iterations for optimization.
        """
        AL = set.X
        cost = -1
        print(f"[Info]: Training Nnet model at lr {learning_rate} for {num_iterations} iterations.")        
        
        for i in range(0, num_iterations):
            # Forward propagation.
            AL = self.model_forward(set.X)

            # Calculate cost.
            cost = calc_cost(AL, set.Y)

            # Backward propagation.
            grads = self.model_backward(AL, set.Y)

            # Update all of the parameters with the newly gained output.
            self.update_layers(grads, learning_rate)

            if log and i % 100 == 0:
                print(f"[Info]: Training: Cost {cost} on iteration {i+1}/{num_iterations}")

        print(f"[Info]: Finished training model with final cost {cost}.")
    
    ##@jit(forceobj=True)
    def model_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation step for a n-level neural network. This network
        is assumed to use RELU nodes up to the output node, which is sigmoid.
        
        :param X: Input layer
        :param nnet: Network layer weights and biases.
        :returns list: Returns a list of the output activation value and the
            cached activation values.
        """

        caches = [] # type: list[ActivationStruct]
        A = X
        L = self.get_layer_count()

        for layer in range(0, L):
            A_prev = A
            current_layer = self.get_layer(layer)
            activation = sim_layer(A_prev, current_layer.W, current_layer.b, current_layer.activation)
            A = activation.A
            caches.append(activation)

        # Store the cached activation data for the backprop stage.
        self.last_iter_caches = caches

        return A
    

    ##@jit()
    def model_backward(self, AL: np.ndarray, Y: np.ndarray) -> list[ActivationDerivatives]:
        """
        Backward propagation step for optimizing costs.
        
        :param AL: Final probability output.
        :param Y: Label vector (whether the thing is or isn't what was predicted).
        :return LayerGradients: Output gradients for the costs and derivatives.
        """
        caches = self.last_iter_caches
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        grads = []  # type: list[ActivationDerivatives]

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL

        for l in range(L-1, -1, -1):
            cur_layer = self.get_layer(l)
            current_cache = caches[l]
            derivatives = linear_activation_backward(dA, current_cache, activation_func_lut[cur_layer.activation])
            dA = derivatives.dA_prev
            grads.append(derivatives)

        grads.reverse()
        return grads
    

    def update_layers(self, grads: list[ActivationDerivatives], learning_rate: float) -> None:
        """
        Update the layer weights and parameters.

        :param grads: Input gradients to update.
        """
        L = len(grads)

        for l in range(L):
            cur_cache = grads[l]
            self.get_layer(l).update(cur_cache, learning_rate)



##@jit(forceobj=True)
def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> LinearOutput:
    """
    Implements linear forward propagation stage.
    
    :param A: Last layer's activation data.
    :param W: Layer weights.
    :param b: Layer biases
    :return LinearOutput: Linear output data.
    """
    Z = W.dot(A) + b
    return LinearOutput(Z, A, W, b)


##@jit(forceobj=True)
def sim_layer(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, afunc: Callable) -> ActivationStruct:
    """
    Implements the essential forward propagation step.
    
    :param A_prev: Previous layer's activations.
    :param W: Weights for this layer.
    :param b: Bias vector for this layer
    :param afunc: Activation function to use.
    :return ActivationStruct: The activation data for this layer.
    """

    linear_out = linear_forward(A_prev, W, b)
    activation_data = afunc(linear_out.Z)
    activation_data.add_linear(linear_out)
    return activation_data


#@jit(forceobj=True)
def calc_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    """
    Cost calculation for optimization.

    :param AL: Final activation vector.
    :param Y: Actual result.
    :return cost: Calculated cost of the operation.
    """

    m = Y.shape[1]

    recip_m = (1. / m)
    c1 = -1. * np.dot(Y, np.log(AL).T)
    c2 = np.dot(1 - Y, np.log((1 - AL).T))
    c3 = recip_m * (c1 - c2)

    cost = np.squeeze(c3)
    return cost


##@jit()
def linear_backward(dZ: np.ndarray, activation_cache: ActivationStruct) -> ActivationDerivatives:
    """
    Linear back propagation. Figure out how to optimize this branch.
    
    :param dZ: The derivative of the output.
    :param activation_cache: Tuple of activation data.
    :return ActivationDerivatives: Derivative output data.
    """

    lcache = activation_cache.linear_cache
    A_prev = lcache.cache_A
    W = lcache.cache_W
    #b = lcache.cache_b
    m = A_prev.shape[1]

    recip_m = 1. / m
    dW = recip_m * np.dot(dZ, A_prev.T)
    db = recip_m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return ActivationDerivatives(dA_prev, dW, db)


def linear_activation_backward(dA: np.ndarray, activation_cache: ActivationStruct, abfunc: Callable) -> ActivationDerivatives:
    """
    Backward propagation for the activation layer.
    
    :param dA: Derivative of the last input.
    :param activation_cache: Cached activation values for nodes of this layer.
    :param abfunc: Backward function for activation.
    :return ActivationDerivatives: The derivatives for the next layers.
    """

    dZ = abfunc(dA, activation_cache.Z)
    grads = linear_backward(dZ, activation_cache)
    return grads


#numba.jit_module(forceobj=True)