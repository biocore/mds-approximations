from warnings import warn

import numpy as np
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

        distance_matrix = np.array(distance_matrix)

        try:
            # Perform eigendecomposition in eigsh shift-inter mode ('LM' -
            # largest magnitude eigenvalue finder near sigma=0)
            # to avoid no convergence that sometimes occurs for large randomly
            # generated distance matrices using the 'SM' finder mode
            # (smallest magnitude eigenvalue finder)
            eigenvalues, eigenvectors = eigsh(distance_matrix,
                                              k=num_dimensions_out,
                                              return_eigenvectors=True,
                                              sigma=0,
                                              which='LM')
        except ArpackNoConvergence as e:
            warn('eigsh unable to converge, only returning {} currently '
                 'converged eigenvectors and {} eigenvalues:\n {}\n\n'
                 .format(len(e.eigenvectors), len(e.eigenvalues), str(e)),
                 RuntimeWarning)
            # since there are missing eigenvalues and vectors when ARPACK
            # does not converge, we need to fill in the missing values with
            # NaN so these results can be serialized in a later processing step
            # without throwing an exception
            eigenvectors = e.eigenvectors
            eigenvalues = e.eigenvalues
            eigenvectors, eigenvalues = \
                self._impute_eigendecomposition_results(
                    eigenvectors, eigenvalues, distance_matrix.shape[0])

        return eigenvectors, eigenvalues

    @staticmethod
    def _impute_eigendecomposition_results(eigenvectors, eigenvalues,
                                           expected_num_results):
        """
            Fills in expected_num_results number of NaNs in eigenvalues array;
            and generates expected_num_results number of arrays of NaNs to
            append to eigenvectors array.

        """
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.array(eigenvectors)

        # impute missing eigenvectors, composed of NaN entries
        nan_eigenvecs = np.array([[np.nan] * eigenvectors.shape[1]]
                                 * (expected_num_results
                                    - len(eigenvectors)))
        eigenvectors = np.concatenate((eigenvectors, nan_eigenvecs),
                                      axis=0)

        # impute missing eigenvalues as NaNs
        nan_eigenvals = [[np.nan] * (expected_num_results
                                     - len(eigenvalues))]
        eigenvalues = np.append(eigenvalues, nan_eigenvals)
        return eigenvectors, eigenvalues
