from scipy.sparse.linalg import eigsh

from mdsa.algorithm import Algorithm

"""
 Perform matrix decomposition using eigsh routine.
"""


class Eigsh(Algorithm):
    def __init__(self):
        super(Eigsh, self).__init__(algorithm_name='eigsh')

    def run(self, distance_matrix, num_dimensions_out=10):
        super(Eigsh, self).run(distance_matrix, num_dimensions_out)

        eigenvalues, eigenvectors = eigsh(distance_matrix,
                                          k=num_dimensions_out)

        return eigenvectors, eigenvalues
