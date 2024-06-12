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

def L_layer_model_exec():
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.005, num_iterations = 1, print_cost = False)

    print("Cost after first iteration: " + str(costs[0]))

def main(argv: list) -> int:
    #train_x, train_y, layer_dims = load_data()
    network = init_network.init_network([50000, 20, 10, 1])
    return 0


if __name__ == "__main__":
    # Train the model since this is not being executed as a library.
    sys.exit(main(sys.argv))