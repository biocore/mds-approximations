# ----------------------------------------------------------------------------
# Copyright (c) 2015--, mdsa development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

from unittest import TestCase, main
from mdsa.lib import return_forty_two

class LibraryTests(TestCase):
    def test_return_forty_two(self):
        self.assertEqual(return_forty_two(), 42)

if __name__ == '__main__':
    main()
