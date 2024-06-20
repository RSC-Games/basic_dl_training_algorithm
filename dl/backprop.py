import numpy as np
from numba import jit, njit
import numba

from dl.data_structures import ActivationStruct
from dl.activation_functions import sigmoid, relu


@njit()
def sigmoid_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Implements backward propagation on a sigmoid unit.
    
    :param dA: Post activation gradient
    :param activationData: Stored activation values
    :returns float: The derivative of this sigmoid unit
    """

    s = 1. / (1. + np.exp(-Z))
    dZ = dA * s * (1. - s)
    return dZ


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Implements backward propagation on a relu unit.
    
    :param dA: Post activation gradient
    :param activationData: Stored activation values
    :returns float: The derivative of this relu unit.
    """

    dZ = dA
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# Allow looking up the corresponding activation backward function.
activation_func_lut = {sigmoid: sigmoid_backward, relu: relu_backward}