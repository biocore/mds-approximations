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


if __name__ == '__main__':
    unittest.main()
