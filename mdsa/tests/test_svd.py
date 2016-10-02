import numpy as np
import unittest

from skbio.util import get_data_path

from mdsa.algorithms.svd import SVD


class TestSVD(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.svd = SVD()

    def test_svd(self):
        """
        Ensure eigenvalues and eigenvectors are returned.
        """
        eigenvectors, eigenvalues = self.svd.run(self.test_matrix)

        self.assertEqual(eigenvectors.shape, (100, 100))
        self.assertEqual(eigenvalues.shape, (100,))


if __name__ == '__main__':
    unittest.main()
