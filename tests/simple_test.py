import unittest

import numpy as np


class TestSimple(unittest.TestCase):
    def test_simple(self):
        x = np.array([1])
        assert x == x
