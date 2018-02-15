import unittest

import numpy as np
from skbio.util import get_data_path

from mdsa.algorithms.eigsh import Eigsh


class TestEigsh(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.eigsh = Eigsh()

    def test_eigsh(self):
        """
        Ensure eigenvalues and eigenvectors are returned.
        """
        DIMENSIONS_OUT = 10
        eigenvectors, eigenvalues = self.eigsh.run(self.test_matrix,
                                                   DIMENSIONS_OUT)

        self.assertEqual(eigenvectors.shape, (100, DIMENSIONS_OUT))
        self.assertEqual(eigenvalues.shape, (DIMENSIONS_OUT,))

    def test_eigsh_impute(self):
        dm = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        dm_samples = dm.shape[0]

        evec = np.array([[1, 2, 3], [1, 2, 3]])
        evals = np.array([1, 2, 3])

        evec, evals = self.eigsh._impute_eigendecomposition_results(evec,
                                                                    evals,
                                                                    dm_samples)

        expected_evec = np.array([[1., 2., 3.], [1., 2., 3.],
                                  [np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan]])

        expected_evals = np.array([1., 2., 3., np.nan])

        self.assertTrue(evec.tolist(), expected_evec.tolist())
        self.assertTrue(evals.tolist(), expected_evals.tolist())


if __name__ == '__main__':
    unittest.main()
