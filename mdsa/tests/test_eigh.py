import numpy as np
import unittest

from skbio.util import get_data_path

from mdsa.algorithms.eigh import Eigh


class TestEigh(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.eigh = Eigh()

    def test_svd(self):
        """
        Ensure eigenvalues and eigenvectors are returned.
        """
        eigenvectors, eigenvalues = self.eigh.run(self.test_matrix)

        self.assertEqual(eigenvectors.shape, (100, 100))
        self.assertEqual(eigenvalues.shape, (100,))


if __name__ == '__main__':
    unittest.main()
