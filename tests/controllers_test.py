import unittest

import numpy as np


class TestController1(unittest.TestCase):
    def test_1(self):
        x = np.array([1])
        assert x == x
