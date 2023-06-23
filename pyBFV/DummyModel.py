import numpy as np


class DummyModel:
    """
    Generates a polynomial 1D model which takes as input a vector and returns the polynomial evaluated in the
    provided input coordinates
    """

    def __init__(self, coefficients):
        """

        :param coefficients: Polynomial coefficients from grade 0 to grade n
        """
        self.coefficients = np.array(coefficients)
        self.degree = len(coefficients) - 1

    def generateOutput(self, x, with_noise=False, std=0, mean=0):
        x = np.array(x)
        exp_vector = np.arange(0, self.degree + 1)
        exp_matrix = np.full((x.shape[0], self.degree + 1), exp_vector)
        x_matrix = np.full((self.degree + 1, x.shape[0]), x).T
        poly_matrix = np.power(x_matrix, exp_matrix)

        y = poly_matrix.dot(self.coefficients.T)

        if with_noise:
            return y + np.random.normal(mean, std, y.shape)
        else:
            return y
