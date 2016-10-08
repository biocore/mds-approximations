import numpy as np
import unittest

from skbio.util import get_data_path

from mdsa.algorithms.ssvd import SSVD


class TestSSVD(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.ssvd = SSVD()

    def test_svd(self):
        """
        Ensure eigenvalues and eigenvectors are returned.
        """
        DIMENSIONS_OUT = 10
        eigenvectors, eigenvalues = self.ssvd.run(self.test_matrix,
                                                  DIMENSIONS_OUT)

        self.assertEqual(eigenvectors.shape, (100, DIMENSIONS_OUT))
        self.assertEqual(eigenvalues.shape, (DIMENSIONS_OUT,))


if __name__ == '__main__':
    unittest.main()
