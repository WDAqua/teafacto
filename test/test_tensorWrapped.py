from unittest import TestCase
from teafacto.core.base import Val, tensorops as T
import numpy as np



class TestTensorWrapped(TestCase):
    def test_dimswap(self):
        xval = np.random.randint(0, 5, (2, 3, 4))
        x = Val(xval)
        y = x.dimswap(1, 0)
        eyval = xval.transpose(1, 0, 2)
        self.assertTrue(np.allclose(eyval, y.eval()))

    def test_reverse(self):
        xval = np.random.randint(0, 5, (2, 3, 4))
        x = Val(xval)
        y = x.reverse(1)
        yval = xval[:, ::-1, :]
        self.assertTrue(np.allclose(yval, y.eval()))
        y = x.reverse(0, 1)
        yval = xval[::-1, ::-1, :]
        self.assertTrue(np.allclose(yval, y.eval()))
