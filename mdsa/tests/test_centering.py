import copy
import unittest

import numpy as np
from skbio.stats.ordination._utils import f_matrix, e_matrix
from skbio.util import get_data_path

from mdsa.centering import center_distance_matrix


class TestCentering(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.test_matrix_copy = copy.deepcopy(self.test_matrix)

    def test_centering(self):
        """
        Ensure optimized centering code outputs a distance matrix
        equal to the output of unoptimized centering code
        already existing in skbio.
        """
        centered_matrix = center_distance_matrix(self.test_matrix)

        # Use deep-copied test matrix because self.test_matrix will have
        # been mutated
        skbio_matrix = self._skbio_center_distance_matrix(
            self.test_matrix_copy)

        self.assertEqual(skbio_matrix.shape, centered_matrix.shape)
        self.assertTrue(np.allclose(skbio_matrix, centered_matrix))

    @staticmethod
    def _skbio_center_distance_matrix(matrix):
        return f_matrix(e_matrix(matrix))


if __name__ == '__main__':
    unittest.main()
