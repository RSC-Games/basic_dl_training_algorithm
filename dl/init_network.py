import numpy as np
import h5py
from dl.data_structures import NeuralNetwork, DataSet


def init_network(layer_dims: list[int]) -> NeuralNetwork:
    """
    Creates a new neural network with the given layer sizes.
    
    :param layer_dims: Layer dimensions (first layer being the input layer).
    :return NeuralNetwork: The created network.
    """
    print(f"[Info]: Creating neural network with layer sizes {layer_dims}")
    return NeuralNetwork(layer_dims)


def load_dataset() -> tuple[DataSet, DataSet, np.ndarray]:
    print("[Warn]: Data loading is fixed function and inflexible!")
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # type: ignore # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # type: ignore # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # type: ignore # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # type: ignore # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # type: ignore # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return DataSet(train_set_x_orig, train_set_y_orig), DataSet(test_set_x_orig, test_set_y_orig), classes
