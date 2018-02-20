import unittest

from skbio import OrdinationResults
from skbio.util import get_data_path

from mdsa.procrustes import _reorder_rows, _pad_matrix, _pad_matrices, \
    procrustes


class TestProcrustes(unittest.TestCase):
    def setUp(self):
        self.test_matrix = OrdinationResults.read(
            get_data_path('unweighted_unifrac_pc.txt'))

    def test_procrustes(self):
        m_squared = procrustes(self.test_matrix, self.test_matrix)

        self.assertAlmostEqual(m_squared, 0)


def test_reorder_rows(self):
    sample_ids_a = [1, 2, 3]
    sample_ids_b = [3, 2, 1, 0]

    order = list(set(sample_ids_a) & set(sample_ids_b))

    # sanity check: expected contents of our order array for testing
    assert order == [1, 2, 3]

    samples = [[0.0, 0.1],
               [1.0, 1.1],
               [2.0, 2.1],
               [3.0, 3.1]]
    res = _reorder_rows(samples, sample_ids_b, order)

    ref = [[2.0, 2.1],
           [1.0, 1.1],
           [0.0, 0.1]]

    self.assertEqual(ref, res.tolist())


def test_pad_matrix(self):
    a = [[1, 2], [1, 2]]

    result_1 = _pad_matrix(a, 2)
    result_2 = _pad_matrix(a, 0)

    ref_1 = [[1, 2, 0, 0], [1, 2, 0, 0]]
    ref_2 = [[1, 2], [1, 2]]

    self.assertEqual(ref_1, result_1.tolist())

    self.assertEqual(ref_2, result_2.tolist())


def _test_pad_matrices(self):
    a = [[1, 2], [1, 2]]
    b = [[1], [1]]
    res_a, res_b = _pad_matrices(a, b)

    ref_a = [[1, 2], [1, 2]]
    ref_b = [[1, 0], [1, 0]]

    self.assertEqual(ref_a, res_a.tolist())
    self.assertEqual(ref_b, res_b.tolist())


if __name__ == '__main__':
    unittest.main()
