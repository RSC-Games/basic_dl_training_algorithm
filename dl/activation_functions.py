from numba import jit
import numba
import numpy as np

from data_structures import ActivationStruct


@jit(numba.float32(numba.float32))
def sigmoid(Z: np.ndarray) -> ActivationStruct:
    """
    Implements the sigmoid activation function.

    :param z: Input z value.
    :returns float: Output activation value.
    """

    A = 1. / (1. + np.exp(-Z))
    return ActivationStruct(A, Z)


def tanh(Z: np.ndarray) -> ActivationStruct:
    """
    Implements the tanh activation function.

    :param z: Input z value.
    :returns float: Output activation value.
    """

    #A = tanh()
    raise NotImplementedError("Unimplemented function 'tanh'!")


@jit(numba.float32(numba.float32))
def relu(Z : np.ndarray) -> ActivationStruct:
    """
    Implements the relu activation function.

    :param z: Input z value.
    :returns float: Output activation value.
    """

    A = np.maximum(0, Z)
    return ActivationStruct(A, Z)