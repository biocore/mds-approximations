import unittest

import numpy as np
from numpy import matrix
from skbio import DistanceMatrix
from skbio.util import get_data_path

from mdsa.algorithms.scmds import Scmds, CombineMds


class TestSCMDS(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))
        self.dist_func = lambda x, y: (self.test_matrix[x, y])
        self.num_objects = self.test_matrix.shape[0]
        self.scmds = Scmds()

    def test_scmds(self):
        # wrap self.test_matrix in an skbio distance matrix
        distance_matrix = DistanceMatrix(self.test_matrix, [str(id_) for id_ in xrange(self.test_matrix.shape[0])])
        self.scmds.run(distance_matrix)

    def test_scmds_output_dimensions(self):
        """ Ensure that an incorrect num_dimensions_out parameter triggers an exception """
        with self.assertRaises(ValueError):
            self.scmds.run(self.test_matrix, num_dimensions_out=self.test_matrix.shape[0])

    def test_scmds_cmds_tzeng(self):
        """cmds_tzeng() should return eigenvectors and eigenvalues,
        sorted by the eigenvalues
        """

        dim = 3
        (eigvec, eigval) = self.scmds._cmds_tzeng(self.test_matrix, dim)
        self.assertTrue(len(eigval) == dim)
        self.assertTrue(eigvec.shape == (self.test_matrix.shape[0], dim))
        self.assertTrue(sorted(eigval, reverse=True) == eigval.tolist())

        self.assertAlmostEqual(eigval[0], 27336.883436, places=5)
        self.assertAlmostEqual(eigval[-1], 536.736247, places=5)
        self.assertAlmostEqual(eigvec[0, 0], -14.978621, places=5)
        self.assertAlmostEqual(eigvec[-1, -1], 0.673001, places=5)

        # full pcoa
        dim = self.test_matrix.shape[0]
        (eigvec, eigval) = self.scmds._cmds_tzeng(self.test_matrix, dim)
        self.assertTrue(len(eigval) == dim)
        self.assertTrue(eigvec.shape == (self.test_matrix.shape[0], dim))
        self.assertTrue(sorted(eigval, reverse=True) == eigval.tolist())

        self.assertAlmostEqual(eigval[0], 27336.883436043929, places=5)
        self.assertAlmostEqual(eigval[-1], 0.000000, places=5)
        self.assertAlmostEqual(eigvec[0, 0], -14.978621, places=5)
        self.assertAlmostEqual(eigvec[-1, -1], 0.000000, places=5)

    def test_scmds_rowmeans(self):
        """rowmeans() should return a vector of row-means for a 2d matrix.
        """

        rm = self.scmds.combine_mds.rowmeans(self.test_matrix[:10])
        self.assertTrue(rm.shape[0] == 10)
        self.assertAlmostEqual(float(rm[0]), 25.983320, places=5)
        self.assertAlmostEqual(float(rm[-1]), 23.967050, places=5)

    def test_scmds_recenter(self):
        """recenter() should recenter an mds solution
        """

        mds_coords = matrix([
            [-3.78333558, -2.90925004, 2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526, 3.99102424, 0.86289149],
            [4.23608631, 0.7933192, 1.16523198]])

        centered_coords = self.scmds.combine_mds.recenter(mds_coords)
        self.assertTrue(centered_coords.shape == mds_coords.shape)
        center_of_gravity = sum([centered_coords[:, x].sum()
                                 for x in range(centered_coords.shape[1])])
        self.assertAlmostEqual(center_of_gravity, 0.0, places=5)

    def test_scmds_affine_mapping(self):
        """affine_mapping() should return a touple of two matrices of
        certain shape
        """

        matrix_x = matrix([
            [-24.03457111, 10.10355666, -23.17039728, 28.48438894,
             22.57322482],
            [0.62716392, 20.84502664, 6.42317521, -7.66901011,
             14.37923852],
            [-2.60793417, 2.83532649, 2.91024821, 1.37414959,
             -4.22916659]])
        matrix_y = matrix([
            [-29.81089477, -2.01312927, -30.5925487, 23.05985801,
             11.68581751],
            [-4.68879117, 23.62633294, 1.07934315, 0.57461989,
             20.52800221],
            [-0.51503505, 1.18377044, 0.83671471, 1.81751358,
             -3.28812925]])

        dim = matrix_x.shape[0]

        (tu, tb) = self.scmds.combine_mds.affine_mapping(matrix_x, matrix_y)
        self.assertTrue(tu.shape[0] == dim and tb.shape[0] == dim)
        self.assertTrue(tu.shape[0] == tu.shape[1])
        self.assertAlmostEqual(tu[0, 0], 0.966653, places=5)
        self.assertAlmostEqual(tu[-1, -1], 0.994816, places=5)
        self.assertAlmostEqual(tb[0, 0], -6.480975, places=5)
        self.assertAlmostEqual(tb[-1, -1], 0.686521, places=5)

    def test_scmds_adjust_mds_to_ref(self):
        """adjust_mds_to_ref() should return an adjusted mds solutions"""

        overlap = 5
        dim = 3
        size = 10
        fake_mds_coords_ref = self.test_matrix[:size, :dim]
        fake_mds_coords_add = self.test_matrix[overlap:size + overlap, :dim]

        mds_adj = self.scmds.combine_mds.adjust_mds_to_ref(fake_mds_coords_ref,
                                    fake_mds_coords_add, overlap)
        self.assertTrue(mds_adj.shape == fake_mds_coords_add.shape)
        self.assertAlmostEqual(mds_adj[0, 0], 7.526609, places=5)
        self.assertAlmostEqual(mds_adj[-1, -1], 18.009350, places=5)

    def test_scmds_combine_mds(self):
        """combine_mds() should merge two mds solutions
        """

        overlap = 3
        dim = 3
        mds_coords_1 = matrix([
            [-3.78333558, -2.90925004, 2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526, 3.99102424, 0.86289149],
            [4.23608631, 0.7933192, 1.16523198]])
        mds_coords_2 = matrix([
            [-3.78333558, -2.90925004, 2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526, 3.99102424, 0.86289149],
            [4.23608631, 0.7933192, 1.16523198]])

        comb_mds = self.scmds.combine_mds.combine_mds(mds_coords_1,
                               mds_coords_2, overlap)
        self.assertTrue(comb_mds.shape == (
            mds_coords_1.shape[0] * 2 - overlap, dim))
        self.assertAlmostEqual(comb_mds[0, 0], -3.783335, places=5)
        # self.assertAlmostEqual(comb_mds[-1, -1], 0.349951)

    def test_scmds_class_combinemds(self):
        """class CombineMds() should be able to join MDS solutions
        """

        # tmp note
        # tile1 = FULL_SYM_MATRIX[0:5, 0:5]
        # tile2 = FULL_SYM_MATRIX[2:7, 2:7]
        # mds_coords_1 = cmds_tzeng(tile1, 3)
        # mds_coords_2 = cmds_tzeng(tile2, 3)
        overlap = 3
        mds_coords_1 = matrix([
            [-3.78333558, -2.90925004, 2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526, 3.99102424, 0.86289149],
            [4.23608631, 0.7933192, 1.16523198]])
        mds_coords_2 = matrix([
            [-3.78333558, -2.90925004, 2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526, 3.99102424, 0.86289149],
            [4.23608631, 0.7933192, 1.16523198]])

        comb_mds = CombineMds()
        comb_mds.add(mds_coords_1, overlap)
        comb_mds.add(mds_coords_2, overlap)
        final_mds = comb_mds.getFinalMDS()
        r, c = mds_coords_1.shape
        self.assertTrue(final_mds.shape == (r * 2 - overlap, c))
        # self.assertAlmostEqual(final_mds[0, 0], 0.0393279)
        # self.assertAlmostEqual(final_mds[-1, -1], -5.322599)


if __name__ == '__main__':
    unittest.main()
