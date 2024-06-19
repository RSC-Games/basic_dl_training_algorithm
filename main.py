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

def main(argv: list) -> int:
    train_set, test_set, classes = init_network.load_dataset()

    network = init_network.init_network([50000, 20, 10, 1])
    network.train()
    return 0


if __name__ == "__main__":
    # Train the model since this is not being executed as a library.
    sys.exit(main(sys.argv))