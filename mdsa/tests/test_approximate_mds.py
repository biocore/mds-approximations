from unittest import TestCase, main
from mdsa import goodness_of_fit
from mdsa.approximate_mds \
    import nystrom
from mdsa.approximate_mds \
    import calc_matrix_a, calc_matrix_b, build_seed_matrix
from mdsa.approximate_mds import rowmeans, \
    affine_mapping, adjust_mds_to_ref, recenter, combine_mds, \
    cmds_tzeng, CombineMds
from numpy import matrix, random, argsort, loadtxt
from skbio.util import get_data_path

PRINT_STRESS = False


# Bigish symmetrical matrix for testing: The following is a distance
# matrix of 100 points making up a 16-dimensional spiral. Idea was
# copied from Tzeng et al. 2008 (PMID 18394154).
#
# Note: the objects are ordered, i.e. permuting the distances will
# give better MDS approximations
#
FULL_SYM_MATRIX = loadtxt(get_data_path('full_sym_matrix.txt'))


class FastMetricScmdsScalingTests(TestCase):
    """test the functions to do metric scaling"""

    def setUp(self):
        """creates inputs"""

        self.dist_func = lambda x, y: (FULL_SYM_MATRIX[x, y])
        self.num_objects = FULL_SYM_MATRIX.shape[0]

    def test_scmds_cmds_tzeng(self):
        """cmds_tzeng() should return eigenvectors and eigenvalues,
        sorted by the eigenvalues
        """

        dim = 3
        (eigvec, eigval) = cmds_tzeng(FULL_SYM_MATRIX, dim)
        self.assertTrue(len(eigval) == dim)
        self.assertTrue(eigvec.shape == (FULL_SYM_MATRIX.shape[0], dim))
        self.assertTrue(sorted(eigval, reverse=True) == eigval.tolist())

        self.assertAlmostEqual(eigval[0], 27336.883436, places=5)
        self.assertAlmostEqual(eigval[-1], 536.736247, places=5)
        self.assertAlmostEqual(eigvec[0, 0], -14.978621, places=5)
        self.assertAlmostEqual(eigvec[-1, -1], 0.673001, places=5)

        # full pcoa
        dim = FULL_SYM_MATRIX.shape[0]
        (eigvec, eigval) = cmds_tzeng(FULL_SYM_MATRIX, dim)
        self.assertTrue(len(eigval) == dim)
        self.assertTrue(eigvec.shape == (FULL_SYM_MATRIX.shape[0], dim))
        self.assertTrue(sorted(eigval, reverse=True) == eigval.tolist())

        self.assertAlmostEqual(eigval[0], 27336.883436043929, places=5)
        self.assertAlmostEqual(eigval[-1], 0.000000, places=5)
        self.assertAlmostEqual(eigvec[0, 0], -14.978621, places=5)
        self.assertAlmostEqual(eigvec[-1, -1], 0.000000, places=5)

    def test_scmds_rowmeans(self):
        """rowmeans() should return a vector of row-means for a 2d matrix.
        """

        rm = rowmeans(FULL_SYM_MATRIX[:10])
        self.assertTrue(rm.shape[0] == 10)
        self.assertAlmostEqual(float(rm[0]),  25.983320, places=5)
        self.assertAlmostEqual(float(rm[-1]), 23.967050, places=5)

    def test_scmds_recenter(self):
        """recenter() should recenter an mds solution
        """

        mds_coords = matrix([
            [-3.78333558, -2.90925004,  2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526,  3.99102424,  0.86289149],
            [4.23608631,  0.7933192,  1.16523198]])

        centered_coords = recenter(mds_coords)
        self.assertTrue(centered_coords.shape == mds_coords.shape)
        center_of_gravity = sum([centered_coords[:, x].sum()
                                 for x in range(centered_coords.shape[1])])
        self.assertAlmostEqual(center_of_gravity, 0.0, places=5)

    def test_scmds_affine_mapping(self):
        """affine_mapping() should return a touple of two matrices of
        certain shape
        """

        matrix_x = matrix([
            [-24.03457111,  10.10355666, -23.17039728,  28.48438894,
             22.57322482],
            [0.62716392,  20.84502664,   6.42317521,  -7.66901011,
             14.37923852],
            [-2.60793417,   2.83532649,   2.91024821,   1.37414959,
             -4.22916659]])
        matrix_y = matrix([
            [-29.81089477,  -2.01312927, -30.5925487,  23.05985801,
             11.68581751],
            [-4.68879117,  23.62633294,   1.07934315,   0.57461989,
             20.52800221],
            [-0.51503505,   1.18377044,   0.83671471,   1.81751358,
             -3.28812925]])

        dim = matrix_x.shape[0]

        (tu, tb) = affine_mapping(matrix_x, matrix_y)
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
        fake_mds_coords_ref = FULL_SYM_MATRIX[:size, :dim]
        fake_mds_coords_add = FULL_SYM_MATRIX[overlap:size + overlap, :dim]

        mds_adj = adjust_mds_to_ref(fake_mds_coords_ref,
                                    fake_mds_coords_add, overlap)
        self.assertTrue(mds_adj.shape == fake_mds_coords_add.shape)
        self.assertAlmostEqual(mds_adj[0, 0],  7.526609, places=5)
        self.assertAlmostEqual(mds_adj[-1, -1], 18.009350, places=5)

    def test_scmds_combine_mds(self):
        """combine_mds() should merge two mds solutions
        """

        overlap = 3
        dim = 3
        mds_coords_1 = matrix([
            [-3.78333558, -2.90925004,  2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526,  3.99102424,  0.86289149],
            [4.23608631,  0.7933192,  1.16523198]])
        mds_coords_2 = matrix([
            [-3.78333558, -2.90925004,  2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526,  3.99102424,  0.86289149],
            [4.23608631,  0.7933192,  1.16523198]])

        comb_mds = combine_mds(mds_coords_1,
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
            [-3.78333558, -2.90925004,  2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526,  3.99102424,  0.86289149],
            [4.23608631,  0.7933192,  1.16523198]])
        mds_coords_2 = matrix([
            [-3.78333558, -2.90925004,  2.75333034],
            [5.18887751, -1.2130882, -0.86476508],
            [-3.10404298, -0.6620052, -3.91668873],
            [-2.53758526,  3.99102424,  0.86289149],
            [4.23608631,  0.7933192,  1.16523198]])

        comb_mds = CombineMds()
        comb_mds.add(mds_coords_1, overlap)
        comb_mds.add(mds_coords_2, overlap)
        final_mds = comb_mds.getFinalMDS()
        r, c = mds_coords_1.shape
        self.assertTrue(final_mds.shape == (r * 2 - overlap, c))
        # self.assertAlmostEqual(final_mds[0, 0], 0.0393279)
        # self.assertAlmostEqual(final_mds[-1, -1], -5.322599)


class FastMetricNystromScalingTests(TestCase):
    """test the functions to do metric scaling"""

    def setUp(self):
        """creates inputs"""

        self.big_seed_matrix = FULL_SYM_MATRIX[:49]
        self.small_seed_matrix = FULL_SYM_MATRIX[:25]

    def test_calc_matrix_a(self):
        """calc_matrix_a should calculate a k x k matrix of
        (predefined) association matrix K of certain (predefined)
        value"""

        nseeds = self.small_seed_matrix.shape[0]
        matrix_e = self.small_seed_matrix[:, 0:nseeds]
        matrix_a = calc_matrix_a(matrix_e)
        self.assertAlmostEqual(matrix_a[0, 0], 250.032270, places=5)
        self.assertAlmostEqual(matrix_a[-1][-1], 316.875461, places=5)

    def test_nystrom_build_seed_matrix(self):
        """build_seed_matrix() should return a seedmatrix and an order
        """

        seedmat_dim = 10
        (seedmat, order) = build_seed_matrix(
            FULL_SYM_MATRIX.shape[0], seedmat_dim,
            lambda x, y: (FULL_SYM_MATRIX[x, y]))
        self.assertTrue(len(order) == FULL_SYM_MATRIX.shape[0])
        self.assertTrue(sorted(order) == range(FULL_SYM_MATRIX.shape[0]))
        self.assertTrue(seedmat.shape == (
            seedmat_dim, FULL_SYM_MATRIX.shape[0]))

        # build_seed_matrix randomises order
        ind = argsort(order)
        i = random.randint(0, seedmat.shape[0])
        j = random.randint(0, seedmat.shape[1])
        self.assertAlmostEqual(seedmat[i, j], FULL_SYM_MATRIX[ind[i], ind[j]],
                               places=5)

    def test_nystrom(self):
        """nystrom() should return an MDS approximation"""

        dim = 3
        mds_coords = nystrom(self.big_seed_matrix, dim)
        self.assertTrue(len(mds_coords.shape) == 2)
        self.assertTrue(mds_coords.shape[0] == self.big_seed_matrix.shape[1])
        self.assertTrue(mds_coords.shape[1] == dim)
        self.assertAlmostEqual(mds_coords[0, 0], -10.709626, places=5)
        self.assertAlmostEqual(mds_coords[-1, -1], -1.778160, places=5)

    def test_nystrom_seed_number(self):
        """nystrom() should give better MDS approximations the more
        seeds were used"""

        dim = 3
        mds_coords = nystrom(self.big_seed_matrix, dim)
        stress = goodness_of_fit.Stress(FULL_SYM_MATRIX, mds_coords)
        kruskal_stress_big_mat = stress.calcKruskalStress()
        if PRINT_STRESS:
            print("INFO: Kruskal stress for Nystrom MDS "
                  "(big_seed_matrix, dim=%d) = %f" %
                  (dim, kruskal_stress_big_mat))
        self.assertTrue(kruskal_stress_big_mat < 0.04)

        mds_coords = nystrom(self.small_seed_matrix, dim)
        stress = goodness_of_fit.Stress(FULL_SYM_MATRIX, mds_coords)
        kruskal_stress_small_mat = stress.calcKruskalStress()
        if PRINT_STRESS:
            print("INFO: Kruskal stress for Nystrom MDS "
                  "(small_seed_matrix, dim=%d) = %f" %
                  (dim, kruskal_stress_small_mat))
        self.assertTrue(kruskal_stress_small_mat < 0.06)

        self.assertTrue(kruskal_stress_small_mat > kruskal_stress_big_mat)

    def test_calc_matrix_b(self):
        """calc_matrix_b should calculate a k x n-k matrix of
        association matrix K
        """

        nseeds = self.small_seed_matrix.shape[0]
        matrix_e = self.small_seed_matrix[:, 0:nseeds]
        matrix_f = self.small_seed_matrix[:, nseeds:]
        matrix_b = calc_matrix_b(matrix_e, matrix_f)

        self.assertTrue(matrix_b.shape == matrix_f.shape)
        self.assertAlmostEqual(matrix_b[0, 0], -272.711227,
                               places=5)
        self.assertAlmostEqual(matrix_b[-1, -1], -64.898372,
                               places=5)


# run if called from the command line
if __name__ == '__main__':
    main()
