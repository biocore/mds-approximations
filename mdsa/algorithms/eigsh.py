from warnings import warn

from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from mdsa.algorithm import Algorithm

"""
 Perform matrix decomposition using eigsh routine.
"""


class Eigsh(Algorithm):
    def __init__(self):
        super(Eigsh, self).__init__(algorithm_name='eigsh')

    def run(self, distance_matrix, num_dimensions_out=10):
        super(Eigsh, self).run(distance_matrix, num_dimensions_out)

        try:
            eigenvalues, eigenvectors = eigsh(distance_matrix,
                                              k=num_dimensions_out,
                                              return_eigenvectors=True)
        except ArpackNoConvergence as e:
            warn('eigsh unable to converge, only returning {} currently '
                 'converged eigenvectors and {} eigenvalues:\n {}\n\n'
                 .format(len(e.eigenvectors), len(e.eigenvalues), str(e)),
                 RuntimeWarning)
            return e.eigenvectors, e.eigenvalues

        return eigenvectors, eigenvalues
