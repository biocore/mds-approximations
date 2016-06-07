import unittest

import numpy as np
from skbio.util import get_data_path

from mdsa.algorithms.scmds import scmds


class TestSCMDS(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))

    def test_scmds(self):
        scmds(self.test_matrix)

if __name__ == '__main__':
    unittest.main()
