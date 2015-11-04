import numpy as np
import unittest
from mdsa.scmds import scmds

from skbio.util import get_data_path


class TestSCMDS(unittest.TestCase):
    def setUp(self):
        self.test_matrix = np.loadtxt(get_data_path('full_sym_matrix.txt'))

    def test_scmds(self):
        scmds(self.test_matrix)

if __name__=='__main__':
    unittest.main()
