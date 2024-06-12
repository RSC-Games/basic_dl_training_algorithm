import numpy as np

# Contains important activation data
class ActivationStruct:
    def __init__(self, A: np.ndarray, Z: np.ndarray):
        self.A = A
        self.Z = Z