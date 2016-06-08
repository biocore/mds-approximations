import unittest

import numpy as np
from numpy import random, argsort
from skbio import DistanceMatrix
from skbio.util import get_data_path

from mdsa import goodness_of_fit
from mdsa.algorithms.nystrom import Nystrom

PRINT_STRESS = False


class TestNystrom(unittest.TestCase):
    """
    Tests functions to do nystrom fast metric scaling
    """
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.nystrom = Nystrom()
        self.big_seed_matrix = self.test_matrix[:49]
        self.small_seed_matrix = self.test_matrix[:25]

    def test_nystrom(self):
        # wrap self.test_matrix in an skbio distance matrix
        distance_matrix = DistanceMatrix(self.test_matrix, [str(id_) for id_ in xrange(self.test_matrix.shape[0])])
        self.nystrom.run(distance_matrix)

    def test_nystrom_output_dimensions(self):
        """ Ensure that an incorrect num_dimensions_out parameter triggers an exception """
        with self.assertRaises(ValueError):
            self.nystrom.run(self.test_matrix, num_dimensions_out=self.test_matrix.shape[0])

    def test_nystrom_build_seed_matrix(self):
        """build_seed_matrix() should return a seedmatrix and an order
        """

        seedmat_dim = 10
        (seedmat, order) = self.nystrom._build_seed_matrix(
            self.test_matrix.shape[0], seedmat_dim,
            lambda x, y: (self.test_matrix[x, y]))
        self.assertTrue(len(order) == self.test_matrix.shape[0])
        self.assertTrue(sorted(order) == range(self.test_matrix.shape[0]))
        self.assertTrue(seedmat.shape == (
            seedmat_dim, self.test_matrix.shape[0]))

        # build_seed_matrix randomises order
        ind = argsort(order)
        i = random.randint(0, seedmat.shape[0])
        j = random.randint(0, seedmat.shape[1])
        self.assertAlmostEqual(seedmat[i, j], self.test_matrix[ind[i], ind[j]],
                               places=5)

    def test_calc_matrix_a(self):
        """calc_matrix_a should calculate a k x k matrix of
        (predefined) association matrix K of certain (predefined)
        value"""

        nseeds = self.small_seed_matrix.shape[0]
        matrix_e = self.small_seed_matrix[:, 0:nseeds]
        matrix_a = self.nystrom._calc_matrix_a(matrix_e)
        self.assertAlmostEqual(matrix_a[0, 0], 250.032270, places=5)
        self.assertAlmostEqual(matrix_a[-1][-1], 316.875461, places=5)

    def test_calc_matrix_b(self):
        """calc_matrix_b should calculate a k x n-k matrix of
        association matrix K
        """

        nseeds = self.small_seed_matrix.shape[0]
        matrix_e = self.small_seed_matrix[:, 0:nseeds]
        matrix_f = self.small_seed_matrix[:, nseeds:]
        matrix_b = self.nystrom._calc_matrix_b(matrix_e, matrix_f)

        self.assertTrue(matrix_b.shape == matrix_f.shape)
        self.assertAlmostEqual(matrix_b[0, 0], -272.711227,
                               places=5)
        self.assertAlmostEqual(matrix_b[-1, -1], -64.898372,
                               places=5)

    def test_nystrom(self):
        """nystrom() should return an MDS approximation"""

        dim = 3
        mds_coords = self.nystrom._nystrom(self.big_seed_matrix, dim)
        self.assertTrue(len(mds_coords.shape) == 2)
        self.assertTrue(mds_coords.shape[0] == self.big_seed_matrix.shape[1])
        self.assertTrue(mds_coords.shape[1] == dim)
        self.assertAlmostEqual(mds_coords[0, 0], -10.709626, places=5)
        self.assertAlmostEqual(mds_coords[-1, -1], -1.778160, places=5)

    def test_nystrom_seed_number(self):
        """nystrom() should give better MDS approximations the more
        seeds were used"""

        dim = 3
        mds_coords = self.nystrom._nystrom(self.big_seed_matrix, dim)
        stress = goodness_of_fit.Stress(self.test_matrix, mds_coords)
        kruskal_stress_big_mat = stress.calcKruskalStress()
        if PRINT_STRESS:
            print("INFO: Kruskal stress for Nystrom MDS "
                  "(big_seed_matrix, dim=%d) = %f" %
                  (dim, kruskal_stress_big_mat))
        self.assertTrue(kruskal_stress_big_mat < 0.04)

        mds_coords = self.nystrom._nystrom(self.small_seed_matrix, dim)
        stress = goodness_of_fit.Stress(self.test_matrix, mds_coords)
        kruskal_stress_small_mat = stress.calcKruskalStress()
        if PRINT_STRESS:
            print("INFO: Kruskal stress for Nystrom MDS "
                  "(small_seed_matrix, dim=%d) = %f" %
                  (dim, kruskal_stress_small_mat))
        self.assertTrue(kruskal_stress_small_mat < 0.06)

        self.assertTrue(kruskal_stress_small_mat > kruskal_stress_big_mat)

if __name__ == '__main__':
    unittest.main()
