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


def main(argv: list) -> int:
    """
    Train our model.

    :param argv: Input arguments.
    :return int: Return code
    """
    pass


if __name__ == "__main__":
    # Train the model since this is not being executed as a library.
    sys.exit(main(sys.argv))