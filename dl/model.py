from typing import Callable
import numpy as np
from numba import jit
import numba

from dl.data_structures import NeuralNetwork, NetworkActivation, ActivationStruct
from dl.activation_functions import relu, sigmoid


@jit()
def sim_layer(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, afunc: Callable) -> ActivationStruct:
    """
    Implements the essential forward propagation step.
    
    :param A_prev: Previous layer's activations.
    :param W: Weights for this layer.
    :param b: Bias vector for this layer
    :param afunc: Activation function to use.
    :return ActivationStruct: The activation data for this layer.
    """

    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = afunc(Z)


@jit()
def model_forward(X: np.ndarray, nnet: NeuralNetwork) -> tuple[float, NetworkActivation]:
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