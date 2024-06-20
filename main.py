#######################################################################################
#                         Optimized Deep Learning algorithm                           #
#                                                                                     #
#    This algorithm is built, with CUDA and numba, to accelerate learning on small    #
# networks such that their training can be completed in a reasonable amount of time.  #
# This is the test file where we train the algorithm so it is capable of doing what   #
# we want (recognize cats).                                                           #
#                                                                                     #
#######################################################################################
import sys
import time
import numpy as np
from dl import init_network
from dl.model import NeuralNetwork
from dl.data_structures import DataSet
from dl.activation_functions import relu, sigmoid

def main(argv: list) -> int:
    train_set_orig, test_set_orig, classes = init_network.load_dataset()
    train_set = init_network.flatten(train_set_orig)
    test_set = init_network.flatten(test_set_orig)

    # Model-specific initialization.
    train_set.div(255.)
    test_set.div(255.)

    network = init_network.init_network([(12288, relu), (20, relu), (10, relu), (1, sigmoid)])
    train_network_timed(network, train_set)

    print(f"[Info]: Testing train set fit.")
    p_hat = network.predict(train_set)

    print(f"[Info]: Testing test set fit.")
    p_hat = network.predict(test_set)
    return 0


def train_network_timed(network: NeuralNetwork, train_set: DataSet):
    before = time.time()
    network.train(train_set, num_iterations=2500, log=True)
    after = time.time()

    print(f"[Info]: Model took {after - before}s to train.")


if __name__ == "__main__":
    # Train the model since this is not being executed as a library.
    sys.exit(main(sys.argv))