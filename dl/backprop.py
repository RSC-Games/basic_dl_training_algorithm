import numpy as np
from numba import jit
import numba

from dl.data_structures import ActivationStruct


@jit(numba.float32(np.ndarray, ActivationStruct))
def sigmoid_backward(dA: np.ndarray, activationData: ActivationStruct) -> np.ndarray:
    """
    Implements backward propagation on a sigmoid unit.
    
    :param dA: Post activation gradient
    :param activationData: Stored activation values
    :returns float: The derivative of this sigmoid unit
    """

    Z = activationData.Z
    s = 1. / (1. + np.exp(-Z))
    dZ = dA * s * (1. - s)
    return dZ


@jit(numba.float32(np.ndarray, ActivationStruct))
def relu_backward(dA: np.ndarray, activationData: ActivationStruct) -> np.ndarray:
    """
    Implements backward propagation on a relu unit.
    
    :param dA: Post activation gradient
    :param activationData: Stored activation values
    :returns float: The derivative of this relu unit
    """

    Z = activationData.Z
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ