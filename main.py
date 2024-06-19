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
import numpy as np
from dl import init_network
from dl.activation_functions import relu, sigmoid

def main(argv: list) -> int:
    train_set_orig, test_set_orig, classes = init_network.load_dataset()
    train_set = init_network.flatten(train_set_orig)
    test_set = init_network.flatten(test_set_orig)

    # Model-specific initialization.
    train_set.div(255.)
    test_set.div(255.)
    print(test_set.X.shape)

    network = init_network.init_network([(12288, relu), (20, relu), (10, relu), (1, sigmoid)])
    network.train(train_set)
    return 0


if __name__ == "__main__":
    # Train the model since this is not being executed as a library.
    sys.exit(main(sys.argv))