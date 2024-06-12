import numpy as np
from dl.data_structures import NeuralNetwork


def init_network(layer_dims: list[int]) -> NeuralNetwork:
    """
    Creates a new neural network with the given layer sizes.
    
    :param layer_dims: Layer dimensions (first layer being the input layer).
    :return NeuralNetwork: The created network.
    """
    print(f"[Info]: Creating neural network with layer sizes {layer_dims}")
    return NeuralNetwork(layer_dims)