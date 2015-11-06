import numpy as np
import unittest
from mdsa.nystrom import nystrom
from skbio.util import get_data_path


class TestNystrom(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))

    def test_nystrom(self):
        nystrom(self.test_matrix)

if __name__ == '__main__':
    unittest.main()
