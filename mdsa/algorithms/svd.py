import numpy as np
import scipy

from mdsa.algorithm import Algorithm

"""
 Performs svd using a LAPACK (Linear Algebra Package) routine
"""


class SVD(Algorithm):
    def __init__(self):
        super(SVD, self).__init__(algorithm_name='svd')

    def run(self, distance_matrix, num_dimensions_out=None):
        super(SVD, self).run(distance_matrix, num_dimensions_out)

        U, s, Vt = scipy.linalg.svd(distance_matrix, full_matrices=True)

        # eigenvectors (without the imaginary component)
        eigenvectors = np.array(U.real)

        # eigenvalues (without the imaginary component)
        eigenvalues = np.array(s.real)

        return eigenvectors, eigenvalues
