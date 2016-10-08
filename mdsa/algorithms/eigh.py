from scipy.linalg import eigh

from mdsa.algorithm import Algorithm

"""
 Performs matrix factorization using eigh.
"""


class Eigh(Algorithm):
    def __init__(self):
        super(Eigh, self).__init__(algorithm_name='eigh')

    def run(self, distance_matrix, num_dimensions_out=None):
        super(Eigh, self).run(distance_matrix, num_dimensions_out)

        eigenvalues, eigenvectors = eigh(distance_matrix)

        return eigenvectors, eigenvalues
