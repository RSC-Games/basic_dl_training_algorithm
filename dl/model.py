from typing import Callable
import numpy as np
from numba import jit
import numba

from dl.data_structures import ActivationStruct, LinearOutput, ActivationDerivatives
from dl.activation_functions import relu, sigmoid


@jit()
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

    linear_out = linear_forward(A_prev, W, b)
    activation_data = afunc(linear_out.Z)
    return activation_data.add_linear(linear_out)


@jit()
def calc_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    """
    Cost calculation for optimization.

    :param AL: Final activation vector.
    :param Y: Actual result.
    :return cost: Calculated cost of the operation.
    """

    m = Y.shape[1]

    recip_m = (1. / m)
    c1 = -1 * np.dot(Y, np.log(AL.T))
    c2 = np.dot(1 - Y, np.log((1 - AL).T))
    c3 = recip_m * (c1 - c2)

    cost = np.squeeze(c3)
    return cost


@jit()
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
    b = lcache.cache_b
    m = A_prev.shape[1]

    recip_m = 1. / m
    dW = recip_m * np.dot(dZ, A_prev.T)
    db = recip_m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return ActivationDerivatives(dA_prev, dW, db)