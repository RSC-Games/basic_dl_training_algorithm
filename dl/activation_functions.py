from numba import jit
import numba
import numpy as np

from dl.data_structures import ActivationStruct


@jit()
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


@jit()
def relu(Z : np.ndarray) -> ActivationStruct:
    """
    Implements the relu activation function.

    :param z: Input z value.
    :returns float: Output activation value.
    """

    A = np.maximum(0, Z)
    return ActivationStruct(A, Z)