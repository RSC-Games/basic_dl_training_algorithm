from typing import Callable
import numpy as np
import h5py
from dl.data_structures import DataSet
from dl.model import NeuralNetwork
import os
from PIL import Image

def init_network(layer_dims: list[tuple[int, Callable]]) -> NeuralNetwork:
    """
    Creates a new neural network with the given layer sizes.
    
    :param layer_dims: Layer dimensions (first layer being the input layer).
    :return NeuralNetwork: The created network.
    """
    print(f"[Info]: Creating neural network with layer sizes {layer_dims}")
    return NeuralNetwork(layer_dims)


def load_dataset() -> tuple[DataSet, DataSet]:
    print("[Warn]: Data loading is fixed function and inflexible!")
    train_set_x_orig = load_images("train") #not thing for test set
    train_set_y_orig = setType("train") # thing for test set
    test_set_x_orig =load_images("test") # not thing for test set
    test_set_y_orig = setType("test") # thing for test set
    return DataSet(train_set_x_orig, train_set_y_orig), DataSet(test_set_x_orig, test_set_y_orig) # type: ignore

def setType(type):
    folder = './datasets'
    folder += '/' + type
    # List to hold the whether or not image is a dog or not
    y = []
    # Loop through all the files in the folder (ai generated code)
    for i in range(0,2): # loop through each file in both the folders doggo and not_doggo
        folder_path = folder
        if(i == 0):
            folder_path += "/doggos"
        else:
            folder_path += "/not_doggos"
        #loop through each file in the specfied folder
        for file_name in os.listdir(folder_path):
            # if image is dog(is in folder doggos) append 1
            if(i == 0):
                y.append(1)
            #otherwise image is not a dog (append 0)
            else:
                y.append(0)
            
    y = np.array(y)
    y = y.reshape(y.shape[0], 1).T
    print(type)
    print(y.shape)
    return y
def load_images(type):
    folder = './datasets'
    folder += '/' + type
    # List to hold the image data
    images = []
    # Loop through all the files in the folder (ai generated code)
    for i in range(0,2): # Why only 2?
        folder_path = folder
        if(i == 0):
            folder_path += "/doggos"
        else:
            folder_path += "/not_doggos"
        for file_name in os.listdir(folder_path):
            # Construct the full file path
            file_path = folder_path +'/'+ file_name
            image = np.array(Image.open(file_path).resize((128,128)))
            images.append(image)
    images = np.array(images)
    return images

def flatten(dataset: DataSet) -> DataSet:
    return DataSet(
        dataset.X.reshape(dataset.X.shape[0], -1).T,
        dataset.Y
    )
